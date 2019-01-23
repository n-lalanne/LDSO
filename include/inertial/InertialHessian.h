#pragma once
#ifndef VIDSO_INERTIAL_HESSIAN_H_
#define VIDSO_INERTIAL_HESSIAN_H_

using namespace std;

#include "NumTypes.h"

namespace ldso {
	namespace inertial {
		class InertialHessian {
		public:
			inline void setImuToCamTransformation(const SE3 &imuToCam) {
				this->T_CB = imuToCam;
				this->T_BC = imuToCam.inverse();
			}

			inline void setEvalPT(const SO3 &worldDSOToWorld_evalPT) {
				this->R_DW_evalPT = worldDSOToWorld_evalPT;
				this->R_WD_evalPT = worldDSOToWorld_evalPT.inverse();

				this->R_WD_PRE = R_WD_evalPT;
				this->R_DW_PRE = R_DW_evalPT;
			};

			inline void setState(Vec4 x_new) {
				x = x_new;
				this->R_DW_PRE = SO3::exp(x.block<3, 1>(0, 0))* R_DW_evalPT;
				this->R_WD_PRE = this->R_DW_PRE.inverse();
				scale_PRE = scale_evalPT + x[3];

				std::cout << "InertialHessian STATS: " << std::endl;
				std::cout << "R_WD_PRE: " << std::endl << R_WD_PRE.matrix() << std::endl;
				std::cout << "s: " << std::endl << scale_PRE << std::endl;
			}

			SE3 T_BC;
			SE3 T_CB;

			double scale_evalPT = 1;
			double scale_PRE;

			SO3 R_DW_evalPT;
			SO3 R_WD_evalPT;

			SO3 R_WD_PRE;
			SO3 R_DW_PRE;

			//0-2: w, 3: s
			Vec4 x;
			Vec4 x_step;
		private:
		};
	}
}

#endif// VIDSO_INERTIAL_HESSIAN_H_
