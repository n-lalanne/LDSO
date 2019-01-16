#pragma once
#ifndef VIDSO_INERTIAL_FRAME_HESSIAN_H_
#define VIDSO_INERTIAL_FRAME_HESSIAN_H_

using namespace std;

#include "NumTypes.h"
#include "internal/FrameHessian.h"
#include "inertial/PreIntegration.h"

namespace ldso {
	namespace internal {
		class FrameHessian;
	}
	namespace inertial {
		class InertialFrameHessian {
		public:
			shared_ptr<internal::FrameHessian> fromFrameHessian;
			shared_ptr<internal::FrameHessian> toFrameHessian;
		private:
			shared_ptr<inertial::PreIntegration> preIntegration;
		};
	}
}

#endif// VIDSO_INERTIAL_FRAME_HESSIAN_H_