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
			void linearize();
			// start at this frame, i
			shared_ptr<InertialFrameHessian> from;
			// go to this frame, j
			shared_ptr<InertialFrameHessian> to;
			
			// H: start at this frame, i
			Mat1515 H_from;
			// b: start at this frame, i
			Vec15 b_from;

			// H: go to this frame, j
			Mat1515 H_to;
			// b: go to this frame, j
			Vec15 b_to;

		private:
			shared_ptr<inertial::PreIntegration> preIntegration;
		};
	}
}

#endif// VIDSO_INERTIAL_FRAME_FRAME_HESSIAN_H_