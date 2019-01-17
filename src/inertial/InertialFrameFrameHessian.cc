using namespace std;

#include "inertial/InertialFrameHessian.h"

namespace ldso {
	namespace inertial {
		InertialFrameFrameHessian::InertialFrameFrameHessian(shared_ptr<inertial::PreIntegration> preIntegration)
		{
			this->preIntegration = preIntegration;
		}
	}
}
