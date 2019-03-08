#pragma once
#ifndef LDSO_FRAME_H_
#define LDSO_FRAME_H_

#include <vector>
#include <memory>
#include <set>
#include <mutex>

using namespace std;

#include "NumTypes.h"
#include "AffLight.h"
#include "inertial/ImuData.h"

namespace ldso {

    // forward declare
    struct Feature;
    namespace internal {
        class FrameHessian;
    }
    struct Point;

    /**
     * Frame is the basic element holding the pose and image data.
     * Here is only the minimal required data, the inner structures are stored in FrameHessian
     */
    struct Frame {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        Frame();

        /**
         * Constructro with timestamp
         * @param timestamp
         */
        Frame(double timestamp);

        /**
         * this internal structure should be created if you want to use them
         */
        void CreateFH(shared_ptr<Frame> frame, vector<ldso::inertial::ImuData> imuData);

        /**
         * release the inner structures
         * NOTE: call release when you want to actually delete the internal data, otherwise it will stay in memory forever
         * (because the FrameHessian is also holding its shared pointer)
         */
        void ReleaseFH();

        /**
         * release the internal structure in features and map points, call this if you no longer want them
         */
        void ReleaseFeatures();

        /**
         * Release all the internal structures in FrameHessian, PointHessian and immature points
         * called when this frame is marginalized
         */
        inline void ReleaseAll() {
            ReleaseFH();
            ReleaseFeatures();
        }

        /**
         * Set the feature grid
         */
        void SetFeatureGrid();

        /**
         * get the feature indecies around a given point
         * @param x
         * @param y
         * @param radius
         * @return
         */
        vector<size_t> GetFeatureInGrid(const float &x, const float &y, const float &radius);

        /**
         * compute bow vectors
         * @param voc vocabulary pointer
         */
        void ComputeBoW(shared_ptr<ORBVocabulary> voc);

        // get keyframes in window
        set<shared_ptr<Frame>> GetConnectedKeyFrames();

        // get all associated points
        vector<shared_ptr<Point>> GetPoints();

        // save & load
        void save(ofstream &fout);    // this will save all the map points
        void load(ifstream &fin, shared_ptr<Frame> &thisFrame, vector<shared_ptr<Frame>> &allKF);

        // get and write pose
        SE3 getPose() {
            unique_lock<mutex> lck(poseMutex);
            return Tcw;
        }

        void setPose(const SE3 &Tcw) {
            unique_lock<mutex> lck(poseMutex);
            this->Tcw = Tcw;
        }

		// get and write pose
		SE3 getTcwInertial() {
			unique_lock<mutex> lck(poseMutex);
			return TcwInertial;
		}

		SE3 getTbwInertial() {
			unique_lock<mutex> lck(poseMutex);
			return TbwInertial;
		}

		Vec3 getVelocityInertial() {
			unique_lock<mutex> lck(poseMutex);
			return v;
		}

		Vec3 getBiasGyroscopeInertial() {
			unique_lock<mutex> lck(poseMutex);
			return bg;
		}

		Vec3 getBiasAccelerometerInertial() {
			unique_lock<mutex> lck(poseMutex);
			return ba;
		}

		double getScaleInertial() {
			unique_lock<mutex> lck(poseMutex);
			return scale;
		}

		void setPoseInertial(const SE3 &Tcw, double scale, const SE3 &Tbw, const Vec3 &v, const Vec3 bg, const Vec3 ba) {
			unique_lock<mutex> lck(poseMutex);
			this->TcwInertial = Tcw;
			this->scale = scale;
			this->TbwInertial = Tbw;
			this->v = v;
			this->bg = bg;
			this->ba = ba;
		}

        // get and write the optimized pose by loop closing
        Sim3 getPoseOpti() {
            unique_lock<mutex> lck(poseMutex);
            return TcwOpti;
        }

        void setPoseOpti(const Sim3 &Scw) {
            unique_lock<mutex> lck(poseMutex);
            TcwOpti = Scw;
        }

        // =========================================================================================================
        // data
        unsigned long id = 0;        // id of this frame
        static unsigned long nextId;  // next id
        unsigned long kfId = 0;      // keyframe id of this frame

    private:
        // poses
        // access them by getPose and getPoseOpti function
        mutex poseMutex;            // need to lock this pose since we have multiple threads reading and writing them
        SE3 Tcw = SE3(Eigen::Quaterniond::Identity(), Vec3::Zero());           // pose from world to camera, estimated by DSO (nobody wants to touch DSO's backend except Jakob)
        Sim3 TcwOpti;     // pose from world to camera optimized by global pose graph (with scale)

		SE3 TcwInertial = SE3(Eigen::Quaterniond::Identity(), Vec3::Zero());
		SE3 TbwInertial = SE3(Eigen::Quaterniond::Identity(), Vec3::Zero());
		Vec3 v;
		Vec3 bg;
		Vec3 ba;
		double scale;
    public:
        bool poseValid = true;     // if pose is valid (false when initializing)
        double timeStamp = 0;      // time stamp
        AffLight aff_g2l;           // aff light transform from global to local
        vector<shared_ptr<Feature>> features;  // Features contained
        vector<vector<std::size_t>> grid;      // feature grid, to fast access features in a given area
        const int gridSize = 20;                // grid size

        // pose relative to keyframes in the window, stored as T_cur_ref
        // this will be changed by full system and loop closing, so we need a mutex
        std::mutex mutexPoseRel;

        /**
         * Relative pose constraint between key-frames
         */
        struct RELPOSE {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            RELPOSE(Sim3 T = Sim3(), const Mat77 &H = Mat77::Identity(), bool bIsLoop = false) :
                    Tcr(T), isLoop(bIsLoop), info(H) {}

            Sim3 Tcr;    // T_current_reference
            Mat77 info = Mat77::Identity();  // information matrix, inverse of covariance, default is identity
            bool isLoop = false;
        };

        // relative poses within the active window
#ifdef _WIN32
		map<shared_ptr<Frame>, RELPOSE, std::less<shared_ptr<Frame>>, Eigen::aligned_allocator<RELPOSE>> poseRel;
#else
		map<shared_ptr<Frame>, RELPOSE, std::less<shared_ptr<Frame>>, Eigen::aligned_allocator<std::pair<const shared_ptr<Frame>, RELPOSE>>> poseRel;
#endif

        // Bag of Words Vector structures.
        DBoW3::BowVector bowVec;       // BoW Vector
        DBoW3::FeatureVector featVec;  // Feature Vector
        vector<size_t> bowIdx;         // index of the bow-ized corners

        shared_ptr<internal::FrameHessian> frameHessian = nullptr;  // internal data

        // ===== debug stuffs ======= //
        cv::Mat imgDisplay;    // image to display, only for debugging, remain an empty image if setting_show_loopclosing is false
    };

    /**
     * Compare frame ID, used to get a sorted map or set of frames
     */
    class CmpFrameID {
    public:
#ifdef _WIN32
        inline bool operator()(const std::shared_ptr<Frame> &f1, const std::shared_ptr<Frame> &f2) const {
            return f1->id < f2->id;
        }
#else
		inline bool operator()(const std::shared_ptr<Frame> &f1, const std::shared_ptr<Frame> &f2) {
			return f1->id < f2->id;
		}
#endif
    };

    /**
     * Compare frame by Keyframe ID, used to get a sorted keyframe map or set.
     */
    class CmpFrameKFID {
    public:
        inline bool operator()(const std::shared_ptr<Frame> &f1, const std::shared_ptr<Frame> &f2) {
            return f1->kfId < f2->kfId;
        }
    };
}

#endif// LDSO_FRAME_H_
