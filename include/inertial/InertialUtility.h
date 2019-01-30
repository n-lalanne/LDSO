#pragma once
#ifndef VIDSO_INERTIAL_UTILITY_H_
#define VIDSO_INERTIAL_UTILITY_H_

using namespace std;

#include "NumTypes.h"

namespace ldso {
	namespace inertial {
		class InertialUtility {
		public:
			EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

			inline static Sophus::Matrix3d Jr(Sophus::Vector3d omega) {
				double theta = omega.norm();
				double theta2 = theta * theta;
				Sophus::Matrix3d V = SO3::hat(omega);

				if (abs(theta) < Sophus::Constants<double>::epsilon())
					return Sophus::Matrix3d::Identity();
				else
					return Sophus::Matrix3d::Identity() - (1 - cos(theta)) / theta2 * V + (theta - sin(theta)) / (theta2 * theta)*V*V;
			}

			inline static Sophus::Matrix3d JrInv(Sophus::Vector3d omega) {
				double theta = omega.norm();
				double half_theta = theta * 0.5;
				Sophus::Matrix3d V = SO3::hat(omega);
				if (abs(theta) < Sophus::Constants<double>::epsilon())
					return Sophus::Matrix3d::Identity();
				else
					return Sophus::Matrix3d::Identity() + 0.5 * V + (1./(theta*theta) - 0.5* (1+cos(theta))/(theta*sin(theta))) * (V * V);
			}
		};
	}
}

#endif// VIDSO_INERTIAL_UTILITY_H_
