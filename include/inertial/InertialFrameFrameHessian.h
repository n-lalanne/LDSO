#pragma once
#ifndef VIDSO_INERTIAL_FRAME_FRAME_HESSIAN_H_
#define VIDSO_INERTIAL_FRAME_FRAME_HESSIAN_H_

using namespace std;

#include "NumTypes.h"
#include "internal/FrameHessian.h"
#include "inertial/InertialFrameHessian.h"
#include "inertial/PreIntegration.h"

namespace ldso {
	namespace inertial {
		class InertialFrameHessian;

		class InertialFrameFrameHessian {
		public:
			InertialFrameFrameHessian(shared_ptr<inertial::PreIntegration> preIntegration);
			shared_ptr<InertialFrameHessian> from;
			shared_ptr<InertialFrameHessian> to;
		private:
			shared_ptr<inertial::PreIntegration> preIntegration;
		};
	}
}

#endif// VIDSO_INERTIAL_FRAME_FRAME_HESSIAN_H_