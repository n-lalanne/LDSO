#pragma once
#ifndef UTIL_SVD_INVERTER_H_
#define UTIL_SVD_INVERTER_H_

using namespace std;

#include "NumTypes.h"

namespace ldso {
	namespace util {
		class MatrixInverter {
		public:
			EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

			static MatXX invertPosDef(MatXX M, bool fast = true);
		};
	}
}

#endif// UTIL_SVD_INVERTER_H_

