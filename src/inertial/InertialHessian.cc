using namespace std;

#include "inertial/InertialHessian.h"

namespace ldso {
	namespace inertial {
		void InertialHessian::setState(Vec4 x_new) {
			x = x_new;
			this->R_DW_PRE = SO3::exp(x.block<3, 1>(0, 0))* R_DW_evalPT;
			this->R_WD_PRE = this->R_DW_PRE.inverse();
			scale_PRE = scale_evalPT + x[3];

			LOG(INFO) << "Inertial Hessian: omega: [" << R_WD_PRE.log().transpose() << "]; scale: " << scale_PRE;
		}
	}
}
