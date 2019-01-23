using namespace std;

#include "inertial/InertialFrameFrameHessian.h"
#include "inertial/InertialUtility.h"

namespace ldso {
	namespace inertial {
		InertialFrameFrameHessian::InertialFrameFrameHessian(shared_ptr<inertial::PreIntegration> preIntegration)
		{
			this->preIntegration = preIntegration;
		}

		void InertialFrameFrameHessian::linearize()
		{
			r.setZero();
			J_from.setZero();
			J_to.setZero();

			Vec3 g(0, 0, -9.81);

			Vec3 d_dRij_ = preIntegration->d_delta_R_ij_dg * from->db_g_PRE;
			Vec3 dvij_g = to->W_v_B_PRE - from->W_v_B_PRE - g * preIntegration->dt_ij;
			Vec3 dpij_g = to->T_WB_PRE.translation() - from->T_WB_PRE.translation() - (from->W_v_B_PRE + 0.5 * g * preIntegration->dt_ij)*preIntegration->dt_ij;

			SO3 dR_tilde_and_bias_inv = (preIntegration->delta_R_ij*SO3::exp(d_dRij_)).inverse();

			r.block<3, 1>(0, 0) = (dR_tilde_and_bias_inv * from->T_BW_PRE.so3()*to->T_WB_PRE.so3()).log();
			r.block<3, 1>(3, 0) = from->T_BW_PRE.so3()*dvij_g - (preIntegration->delta_v_ij + preIntegration->d_delta_v_ij_dg * from->db_g_PRE + preIntegration->d_delta_v_ij_da*from->db_a_PRE);
			r.block<3, 1>(6, 0) = from->T_BW_PRE.so3() * dpij_g - (preIntegration->delta_p_ij + preIntegration->d_delta_p_ij_dg * from->db_g_PRE + preIntegration->d_delta_p_ij_da*from->db_a_PRE);
			r.block<3, 1>(9, 0) = from->db_g_PRE - to->db_g_PRE;
			r.block<3, 1>(12, 0) = from->db_a_PRE - to->db_a_PRE;

			Mat33 Jr_inv = InertialUtility::JrInv(r.block<3, 1>(0, 0));

			Mat33 dR_dwi = -Jr_inv * to->T_BW_PRE.so3().matrix();

			//dr_R_dw_i
			J_from.block<3, 3>(0, 3) = dR_dwi;
			//dr_R_dbg_i
			J_from.block<3, 3>(0, 9) = -Jr_inv * SO3::exp(r.block<3, 1>(0, 0)).matrix()*InertialUtility::Jr(d_dRij_)*preIntegration->d_delta_R_ij_dg;


			//dr_v_dw_i
			J_from.block<3, 3>(3, 3) = from->T_BW_PRE.so3().matrix() * SO3::hat(dvij_g);
			//dr_v_dv_i
			J_from.block<3, 3>(3, 6) = -from->T_BW_PRE.so3().matrix();
			//dr_v_dbg_i
			J_from.block<3, 3>(3, 9) = -preIntegration->d_delta_v_ij_dg;
			//dr_v_dba_i
			J_from.block<3, 3>(3, 12) = -preIntegration->d_delta_v_ij_da;


			//dr_p_du_i
			J_from.block<3, 3>(6, 0) = -from->T_BW_PRE.so3().matrix();
			//dr_p_dw_i
			J_from.block<3, 3>(6, 3) = from->T_BW_PRE.so3().matrix() * (SO3::hat(dpij_g) + SO3::hat(from->T_WB_PRE.translation()));
			//dr_p_dv_i
			J_from.block<3, 3>(6, 6) = -from->T_BW_PRE.so3().matrix() * preIntegration->dt_ij;
			//dr_p_dbg_i
			J_from.block<3, 3>(6, 9) = -preIntegration->d_delta_p_ij_dg;
			//dr_p_dba_i
			J_from.block<3, 3>(6, 12) = -preIntegration->d_delta_p_ij_da;

			//dr_bg_dbg_i
			J_from.block<3, 3>(9, 9) = Mat33::Identity();

			//dr_ba_dba_i
			J_from.block<3, 3>(12, 12) = Mat33::Identity();


			//dr_R_dw_j
			J_to.block<3, 3>(0, 3) = -dR_dwi;


			//dr_v_dv_j
			J_to.block<3, 3>(3, 6) = from->T_BW_PRE.so3().matrix();


			//dr_p_du_j
			J_to.block<3, 3>(6, 0) = from->T_BW_PRE.so3().matrix();
			//dr_p_dw_j
			J_to.block<3, 3>(6, 3) = -from->T_BW_PRE.so3().matrix()*SO3::hat(to->T_WB_PRE.translation());



			//dr_bg_dbg_j
			J_to.block<3, 3>(9, 9) = -Mat33::Identity();

			//dr_ba_dba_j
			J_to.block<3, 3>(12, 12) = -Mat33::Identity();

			Mat1515 W;
			W.setZero();

			W.block<9, 9>(0, 0) = preIntegration->Sigma_ij;
			W.block<6, 6>(9, 9) = preIntegration->Sigma_bd * preIntegration->dt_ij;

			H_to = J_to.transpose() * W * J_to;
			H_from = J_from.transpose() * W * J_from;

			b_to = -J_to.transpose() * W * r;
			b_from = -J_to.transpose() * W * r;

 			energy = r.transpose() * W * r;
		}
	}
}
