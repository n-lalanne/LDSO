using namespace std;

#include "inertial/PreIntegration.h"
#include "inertial/InertialUtility.h"
#include <Eigen/Geometry> 

namespace ldso {
	namespace inertial {

		double PreIntegration::delta_t = 1.0 / 200.0;

		PreIntegration::PreIntegration()
		{
			lin_bias_g.setZero();
			lin_bias_a.setZero();
			this->reset();
		}

		void PreIntegration::addImuData(vector<ImuData> imuData)
		{
			for (int i = 0; i < imuData.size(); i++)
			{
				ImuData d = imuData[i];
				imuDataHistory.push_back(d);
				integrateOne(d);
			}
		}

		void PreIntegration::reEvaluate()
		{
			reset();
			for (int i = 0; i < imuDataHistory.size(); i++)
			{
				integrateOne(imuDataHistory[i]);
			}
		}

		void PreIntegration::integrateOne(ImuData data)
		{
			Mat99 A;
			Mat96 B;

			A.setZero();
			B.setZero();

			Vec3 omega_corr(data.gx, data.gy, data.gz);
			Vec3 alpha_corr(data.ax, data.ay, data.az);

			omega_corr -= lin_bias_g;
			alpha_corr -= lin_bias_a;

			SO3 ddelta_R_k = SO3::exp(omega_corr * delta_t);
			Mat33 A21 = -delta_R_ij.matrix() * SO3::hat(alpha_corr) * delta_t;

			Mat33 B11 = InertialUtility::Jr(omega_corr * delta_t) * delta_t;
			Mat33 B22 = delta_R_ij.matrix() * delta_t;
			B.block<3, 3>(0, 0) = B11;
			B.block<3, 3>(3, 3) = B22;
			B.block<3, 3>(3, 6) = B22 * delta_t*0.5;

			delta_R_ij = delta_R_ij * ddelta_R_k;
			d_delta_R_ij_dg = SO3::exp(-omega_corr * delta_t).matrix()*d_delta_R_ij_dg - B11;

			Vec3 ddelta_v_ij = delta_R_ij * alpha_corr * delta_t;
			delta_v_ij += ddelta_v_ij;

			Mat33 dd_delta_v_ij_dg = delta_R_ij.matrix() * SO3::hat(alpha_corr)*d_delta_R_ij_dg*delta_t;
			d_delta_v_ij_dg += -dd_delta_v_ij_dg;

			Mat33 dd_delta_v_ij_da = delta_R_ij.matrix() * delta_t;
			d_delta_v_ij_da += -dd_delta_v_ij_da;

			delta_p_ij = delta_p_ij + delta_v_ij * delta_t + 0.5 * ddelta_v_ij * delta_t;
			d_delta_p_ij_dg += d_delta_v_ij_dg * delta_t + 0.5 * dd_delta_v_ij_dg * delta_t;
			d_delta_p_ij_da += d_delta_v_ij_da * delta_t + 0.5 * dd_delta_v_ij_da * delta_t;

			A.block<3, 3>(0, 0) = ddelta_R_k.inverse().matrix();
			A.block<3, 3>(3, 0) = A21;
			A.block<3, 3>(6, 0) = 0.5 * delta_t * A21;

			A.block<3, 3>(3, 3) = Mat33::Identity();
			A.block<3, 3>(6, 3) = Mat33::Identity()*delta_t;
			A.block<3, 3>(6, 6) = Mat33::Identity();

			Sigma_ij = A * Sigma_ij * A.transpose() + B * Sigma_eta * B.transpose();
		}

		void PreIntegration::reset()
		{
			delta_p_ij.setZero();
			delta_v_ij.setZero();
			delta_R_ij.setQuaternion(Eigen::Quaternion<double>(1, 0, 0, 0));

			d_delta_p_ij_dg.setZero();
			d_delta_p_ij_da.setZero();

			d_delta_v_ij_dg.setZero();
			d_delta_v_ij_da.setZero();

			d_delta_R_ij_dg.setZero();

			Sigma_ij.setZero();
		}
	}
}
