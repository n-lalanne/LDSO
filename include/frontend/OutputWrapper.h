#pragma once
#ifndef OUTPUT_WRAPPER_H_
#define OUTPUT_WRAPPER_H_

#include "NumTypes.h"
#include "Frame.h"
#include "Map.h"
#include "frontend/MinimalImage.h"
#include "internal/FrameHessian.h"
#include "internal/PointHessian.h"
#include "internal/CalibHessian.h"
#include "inertial/InertialHessian.h"

using namespace std;

using namespace ldso::internal;

namespace ldso {

	class OutputWrapper {
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

		virtual void publishKeyframes(std::vector<shared_ptr<Frame>> &frames, bool final, shared_ptr<CalibHessian> HCalib, shared_ptr<inertial::InertialHessian> HInertial) {}

		virtual void publishCamPose(shared_ptr<Frame> frame, shared_ptr<CalibHessian> HCalib, shared_ptr<inertial::InertialHessian> HInertial) {}

		virtual void setMap(shared_ptr<Map> m) {}

		virtual void join() {}

		virtual void reset() {}

		virtual void refreshAll() {}
	};

}

#endif // OUTPUT_WRAPPER_H_
