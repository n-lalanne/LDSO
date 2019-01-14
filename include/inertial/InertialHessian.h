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

			inline void setEvalPT(const SO3 &worldDSOToWorld_evalPT, const Vec4 &state) {
				this->R_DW_evalPT = worldDSOToWorld_evalPT;
				this->R_WD_evalPT = worldDSOToWorld_evalPT.inverse();
			};

			EIGEN_STRONG_INLINE const SE3 &get_imuToCam() const {
				return T_CB;
			}

			EIGEN_STRONG_INLINE const SE3 &get_CamToImu() const {
				return T_BC;
			}

			EIGEN_STRONG_INLINE const SO3 &get_worldDSOToWorld_evalPT() const {
				return R_WD_evalPT;
			}

			EIGEN_STRONG_INLINE const SO3 &get_worldDSOToWorld_PRE() const {
				//TODO::
				return R_WD_evalPT;
			}

			EIGEN_STRONG_INLINE const SO3 &get_worldToWorldDSO_PRE() const {
				//TODO::
				return R_WD_evalPT;
			}

		private:
			SE3 T_BC;
			SE3 T_CB;
			double scale_evalPT;
			SO3 R_DW_evalPT;
			SO3 R_WD_evalPT;
		};
	}
}

#endif// VIDSO_INERTIAL_HESSIAN_H_
