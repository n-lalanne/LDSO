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
			shared_ptr<internal::FrameHessian> fh;
			vector<inertial::ImuData> imuDataHistory;
			shared_ptr<inertial::InertialFrameFrameHessian> from;
			shared_ptr<inertial::InertialFrameFrameHessian> to;
		};
	}
}

#endif// VIDSO_INERTIAL_FRAME_HESSIAN_H_