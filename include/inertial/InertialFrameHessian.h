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
			void linearize(shared_ptr<InertialHessian> inertialHessian, double visualWeight);

			void setState(Vec15 x_new);

			Mat2525 H;
			Vec25 b;

			shared_ptr<internal::FrameHessian> fh;
			vector<inertial::ImuData> imuDataHistory;
			// factor starts from me, i
			shared_ptr<inertial::InertialFrameFrameHessian> from;
			// factor goes to me, j
			shared_ptr<inertial::InertialFrameFrameHessian> to;

			Vec3 W_v_B_EvalPT;

			SE3 T_WB_EvalPT;
			SE3 T_BW_EvalPT;


			Vec3 db_a_EvalPT;
			Vec3 db_g_EvalPT;

			Vec3 W_v_B_PRE;

			SE3 T_WB_PRE;
			SE3 T_BW_PRE;

			Vec3 db_a_PRE;
			Vec3 db_g_PRE;

			// 0-2: u, 3-5: w, 6-8: v, 9-11: b_g, 12-14: b_a
			Vec15 x = Vec15::Zero();
			Vec15 x_step = Vec15::Zero();
			Vec15 x_backup = Vec15::Zero();

			Vec6 r;
			Mat625 J;
			double energy;

		};
	}
}

#endif// VIDSO_INERTIAL_FRAME_HESSIAN_H_