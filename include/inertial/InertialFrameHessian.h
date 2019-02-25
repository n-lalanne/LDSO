#pragma once
#ifndef VIDSO_INERTIAL_FRAME_HESSIAN_H_
#define VIDSO_INERTIAL_FRAME_HESSIAN_H_

using namespace std;

#include "NumTypes.h"
#include "inertial/PreIntegration.h"
#include "inertial/InertialFrameFrameHessian.h"
#include "internal/FrameHessian.h"

namespace ldso {
	namespace inernal {
		class FrameHessian;
	}
	namespace inertial {
		class InertialFrameFrameHessian;
		class InertialFrameHessian
		{
		public:
			EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

			void linearize(shared_ptr<InertialHessian> inertialHessian, double visualWeight, bool force, int nPreIntegrationFactors, int nCombineFactors);
			static void computeJacobian(Mat625 &J, double s, SO3 Rwb, SO3 Rbw, SO3 Rcb, SO3 Rbc, SO3 Rcd, SO3 Rdc, SO3 Rdw, SO3 Rwd, Vec3 pw, Vec3 pc, SE3 Tdc);
			static void computeResidual(Vec6 &r, double s, SO3 Rwb, SO3 Rbw, SO3 Rcb, SO3 Rbc, SO3 Rcd, SO3 Rdc, SO3 Rdw, SO3 Rwd, Vec3 pw, Vec3 pc, SE3 Tdc);
			void setState(Vec15 x_new);
			void setCurrentStateAsEvalPt();

			Mat2525 H;
			Vec25 b;

			shared_ptr<internal::FrameHessian> fh;
			vector<inertial::ImuData> imuDataHistory;
			// factor starts from me, i
			shared_ptr<inertial::InertialFrameFrameHessian> from;
			// factor goes to me, j
			shared_ptr<inertial::InertialFrameFrameHessian> to;

			Vec3 W_v_B_EvalPT = Vec3::Zero();

			SE3 T_WB_EvalPT = SE3();

			Vec3 b_g_lin = Vec3::Zero();
			Vec3 b_a_lin = Vec3::Zero();

			Vec3 db_a_EvalPT = Vec3::Zero();
			Vec3 db_g_EvalPT = Vec3::Zero();

			Vec3 W_v_B_PRE = Vec3::Zero();

			SE3 T_WB_PRE = SE3();
			SE3 T_BW_PRE = SE3();

			Vec3 db_a_PRE = Vec3::Zero();
			Vec3 db_g_PRE = Vec3::Zero();

			// 0-2: u, 3-5: w, 6-8: v, 9-11: b_g, 12-14: b_a
			Vec15 x = Vec15::Zero();
			Vec15 x_step = Vec15::Zero();
			Vec15 x_backup = Vec15::Zero();

			Vec6 W;

			Vec6 r;
			Mat625 J;
			double energy;

			double lastVisualWeight;

		};
	}
}

#endif// VIDSO_INERTIAL_FRAME_HESSIAN_H_