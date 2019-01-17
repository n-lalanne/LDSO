using namespace std;

#include "inertial/PreIntegration.h"

namespace ldso {
	namespace inertial {
		void PreIntegration::addImuData(vector<ImuData> imuData)
		{
			for (int i = 0; i < imuData.size(); i++)
			{
				ImuData d = imuData[i];
				imuDataHistory.push_back(d);
				integrateOne(d);
			}
		}

		void PreIntegration::reEvaluate()
		{
			reset();
			for (int i = 0; i < imuDataHistory.size(); i++)
			{
				integrateOne(imuDataHistory[i]);
			}
		}

		void PreIntegration::integrateOne(ImuData data)
		{
		}

		void PreIntegration::reset()
		{

		}
	}
}
