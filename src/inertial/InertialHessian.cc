using namespace std;

#include "inertial/InertialHessian.h"
#include "Settings.h"

namespace ldso {
	namespace inertial {
		void InertialHessian::setState(Vec4 x_new) {
			x = x_new;
			this->R_DW_PRE = SO3::exp(x.block<3, 1>(0, 0))* R_DW_EvalPT;
			this->R_WD_PRE = this->R_DW_PRE.inverse();
			scale_PRE = scale_EvalPT + x[3];

			if (setting_vi_debug)
				LOG(INFO) << "Inertial Hessian: omega: [" << R_WD_PRE.log().transpose().format(setting_vi_format) << "]; scale: " << scale_PRE << " (exp: " << exp(scale_PRE) << ")";
		}
	}
}
