#pragma once
#ifndef VIDSO_PRE_INTEGRATION_H_
#define VIDSO_PRE_INTEGRATION_H_

using namespace std;

#include "NumTypes.h"
#include "inertial/InertialUtility.h"
#include "inertial/ImuData.h"

namespace ldso {
	namespace inertial {
		class PreIntegration {
		public:
			void addImuData(vector<ImuData> imuData);
			void reEvaluate();
		private:
			vector<ImuData> imuDataHistory;
			void reset();
			void integrateOne(ImuData data);
		};
	}
}

#endif// VIDSO_PRE_INTEGRATION_H_

