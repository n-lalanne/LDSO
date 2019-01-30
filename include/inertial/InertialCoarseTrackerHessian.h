#pragma once
#ifndef VIDSO_INERTIAL_COARSE_TRACKER_HESSIAN_H_
#define VIDSO_INERTIAL_COARSE_TRACKER_HESSIAN_H_

using namespace std;

#include "NumTypes.h"
#include "inertial/PreIntegration.h"
#include "internal/FrameHessian.h"
#include "inertial/InertialHessian.h"

namespace ldso {
	namespace inertial {
		class InertialCoarseTrackerHessian
		{
		public:
			EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

			InertialCoarseTrackerHessian();

			void setValues(std::vector <shared_ptr<internal::FrameHessian>> &frameHessians, shared_ptr<inertial::InertialHessian> Hinertial);
			void compute(double visualWeight, SE3 T_id, SE3 T_ji);
			void update(Vec8 x);
			void backup();
			void reset();

			shared_ptr<PreIntegration> preIntegration;

			Vec3 v_i;
			Vec3 v_j;

			SE3 Tw_i;
			SE3 Tw_j;

			Vec3 bg_i;
			Vec3 ba_i;
			Vec3 bg_j;
			Vec3 ba_j;

			Vec15 x_backup_i;
			Vec15 x_backup_j;

			double scale;
			SE3 T_bc;
			SO3 R_wd;

			Vec8 b_I = Vec8::Zero();
			Vec8 b_I_sc = Vec8::Zero();

			Mat88 H_I = Mat88::Zero();
			Mat88 H_I_sc = Mat88::Zero();

			VecX bb = VecX::Zero(15);

			MatXX Hbb_inv = MatXX::Zero(15, 15);
			MatXX Hab = MatXX::Zero(8, 15);

			Mat1515 W;
			Vec6 w;

			Mat2525 S;

			double energy;

			bool fix_i = true;
		};
	}
}

#endif// VIDSO_INERTIAL_COARSE_TRACKER_HESSIAN_H_