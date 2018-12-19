#pragma once
#ifndef VIDSO_IMU_DATA_H_
#define VIDSO_IMU_DATA_H_

using namespace std;

#include "NumTypes.h"

namespace ldso {
	namespace inertial {
		struct ImuData {
			double time;
			double gx;
			double gy;
			double gz;
			double ax;
			double ay;
			double az;
		};
	}
}

#endif// VIDSO_IMU_DATA_H_
