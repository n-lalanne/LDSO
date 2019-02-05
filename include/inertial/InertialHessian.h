#pragma once
#ifndef VIDSO_INERTIAL_HESSIAN_H_
#define VIDSO_INERTIAL_HESSIAN_H_

using namespace std;

#include "NumTypes.h"

namespace ldso {
	namespace inertial {
		class InertialHessian {
		public:
			EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

			inline void setImuToCamTransformation(const SE3 &imuToCam) {
				this->T_CB = imuToCam;
				this->T_BC = imuToCam.inverse();
			}

			inline void setEvalPT(const SO3 &worldToWorldDSO_evalPT) {
				this->R_DW_EvalPT = worldToWorldDSO_evalPT;
				this->R_WD_PRE = R_DW_EvalPT.inverse();
				this->R_DW_PRE = R_DW_EvalPT;
			};

			void setState(Vec4 x_new);
			

			SE3 T_BC;
			SE3 T_CB;

			double scale_EvalPT = 0;
			double scale_PRE;

			SO3 R_DW_EvalPT;

			SO3 R_WD_PRE;
			SO3 R_DW_PRE;

			//0-2: w, 3: s
			Vec4 x = Vec4::Zero();
			Vec4 x_step = Vec4::Zero();
			Vec4 x_backup = Vec4::Zero();
		private:
		};
	}
}

#endif// VIDSO_INERTIAL_HESSIAN_H_
