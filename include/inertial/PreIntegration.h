#pragma once
#ifndef VIDSO_PRE_INTEGRATION_H_
#define VIDSO_PRE_INTEGRATION_H_

using namespace std;

#include "NumTypes.h"
#include "inertial/ImuData.h"

namespace ldso {
	namespace inertial {
		class PreIntegration {
		public:
			EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

			PreIntegration();

			void addImuData(vector<ImuData> imuData);
			void reEvaluate();

			Vec3 lin_bias_g;
			Vec3 lin_bias_a;

			Vec3 delta_p_ij;
			Vec3 delta_v_ij;
			SO3 delta_R_ij;

			Mat33 d_delta_p_ij_dg;
			Mat33 d_delta_p_ij_da;

			Mat33 d_delta_v_ij_dg;
			Mat33 d_delta_v_ij_da;

			Mat33 d_delta_R_ij_dg;

			double dt_ij;

			Mat99 Sigma_ij;

			static Mat66 Sigma_eta;
			static Mat66 Sigma_bd;
			static double delta_t;
		private:
			vector<ImuData> imuDataHistory;
			void reset();
			void integrateOne(ImuData data);
		};
	}
}

#endif// VIDSO_PRE_INTEGRATION_H_

