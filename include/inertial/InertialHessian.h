#pragma once
#ifndef VIDSO_INERTIAL_HESSIAN_H_
#define VIDSO_INERTIAL_HESSIAN_H_

using namespace std;

#include "NumTypes.h"
#include "internal/FrameHessian.h"

namespace ldso {
	namespace internal {
		class FrameHessian;
	}
	namespace inertial {
		class InertialHessian {
		public:
			shared_ptr<internal::FrameHessian> fromFrameHessian;
			shared_ptr<internal::FrameHessian> toFrameHessian;
		private:
		};
	}
}

#endif// VIDSO_INERTIAL_HESSIAN_H_
