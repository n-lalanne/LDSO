#include "Feature.h"
#include "internal/OptimizationBackend/EnergyFunctional.h"
#include "internal/GlobalFuncs.h"

#include "util/MatrixInverter.h"

#include <glog/logging.h>


namespace ldso {

	namespace internal {

		bool EFAdjointsValid = false;
		bool EFIndicesValid = false;
		bool EFDeltaValid = false;

		EnergyFunctional::EnergyFunctional() :
			accSSE_top_L(new AccumulatedTopHessianSSE),
			accSSE_top_A(new AccumulatedTopHessianSSE),
			accSSE_bot(new AccumulatedSCHessianSSE)
		{
			Vec25 s;
			s << SCALE_XI_TRANS, SCALE_XI_TRANS, SCALE_XI_TRANS, SCALE_XI_ROT, SCALE_XI_ROT, SCALE_XI_ROT, SCALE_VI_ROT, SCALE_VI_ROT, SCALE_VI_ROT, SCALE_VI_S, SCALE_VI_TRANS, SCALE_VI_TRANS, SCALE_VI_TRANS, SCALE_VI_ROT, SCALE_VI_ROT, SCALE_VI_ROT, SCALE_VI_V, SCALE_VI_V, SCALE_VI_V, SCALE_VI_B, SCALE_VI_B, SCALE_VI_B, SCALE_VI_B, SCALE_VI_B, SCALE_VI_B;
			S = s.asDiagonal();


			int scaleGravityOffset = 0;

			if (setting_vi_optimize_scale_and_gravity_direction)
				scaleGravityOffset = 4;

			MatXX HM_I = MatXX::Zero(scaleGravityOffset, scaleGravityOffset);
			VecX bM_I = VecX::Zero(scaleGravityOffset);
		}

		EnergyFunctional::~EnergyFunctional() {
			if (adHost != 0) delete[] adHost;
			if (adTarget != 0) delete[] adTarget;
			if (adHostF != 0) delete[] adHostF;
			if (adTargetF != 0) delete[] adTargetF;
			if (adHTdeltaF != 0) delete[] adHTdeltaF;
		}

		void EnergyFunctional::insertResidual(shared_ptr<PointFrameResidual> r) {
			r->takeData();
			connectivityMap[(((uint64_t)r->host.lock()->frameID) << 32) + ((uint64_t)r->target.lock()->frameID)][0]++;
			nResiduals++;
		}

		void EnergyFunctional::insertFrame(shared_ptr<FrameHessian> fh, shared_ptr<CalibHessian> Hcalib) {
			fh->takeData();
			frames.push_back(fh);
			fh->idx = frames.size();
			nFrames++;

			// extend H,b
			assert(HM.cols() == 8 * nFrames + CPARS - 8);
			bM.conservativeResize(8 * nFrames + CPARS);
			HM.conservativeResize(8 * nFrames + CPARS, 8 * nFrames + CPARS);
			bM.tail<8>().setZero();
			HM.rightCols<8>().setZero();
			HM.bottomRows<8>().setZero();

			int scaleGravityOffset = 0;

			if (setting_vi_optimize_scale_and_gravity_direction)
				scaleGravityOffset = 4;

			// extend H,b
			assert(HM_I.cols() == 15 * (nFrames - 1) + scaleGravityOffset);
			bM_I.conservativeResize(15 * nFrames + scaleGravityOffset);
			HM_I.conservativeResize(15 * nFrames + scaleGravityOffset, 15 * nFrames + scaleGravityOffset);
			bM_I.tail<15>().setZero();
			HM_I.rightCols<15>().setZero();
			HM_I.bottomRows<15>().setZero();

			if (fh->frame->kfId == 0)
			{
				HM_I.bottomRightCorner<15, 15>().block<3, 3>(6, 6) = setting_vi_velocity_prior * Vec3::Ones().asDiagonal();
			}

			// set index as invalid
			EFIndicesValid = false;
			EFAdjointsValid = false;
			EFDeltaValid = false;

			setAdjointsF(Hcalib);
			makeIDX();

			// set connectivity map
			for (auto fh2 : frames) {
				connectivityMap[(((uint64_t)fh->frameID) << 32) + ((uint64_t)fh2->frameID)] = Eigen::Vector2i(0, 0);
				if (fh2 != fh)
					connectivityMap[(((uint64_t)fh2->frameID) << 32) + ((uint64_t)fh->frameID)] = Eigen::Vector2i(0,
						0);
			}
		}

		void EnergyFunctional::dropResidual(shared_ptr<PointFrameResidual> r) {

			// remove this residual from pointHessian->residualsAll
			shared_ptr<PointHessian> p = r->point.lock();
			deleteOut<PointFrameResidual>(p->residuals, r);
			connectivityMap[(((uint64_t)r->host.lock()->frameID) << 32) + ((uint64_t)r->target.lock()->frameID)][0]--;
			nResiduals--;
		}

		void EnergyFunctional::marginalizeFrame(shared_ptr<FrameHessian> fh) {

			assert(EFDeltaValid);
			assert(EFAdjointsValid);
			assert(EFIndicesValid);

			marginalizeInertialFrameHessian(fh);

			int ndim = nFrames * 8 + CPARS - 8;// new dimension
			int odim = nFrames * 8 + CPARS;// old dimension

			if ((int)fh->idx != (int)frames.size() - 1) {
				int io = fh->idx * 8 + CPARS;    // index of frame to move to end
				int ntail = 8 * (nFrames - fh->idx - 1);
				assert((io + 8 + ntail) == nFrames * 8 + CPARS);

				Vec8 bTmp = bM.segment<8>(io);
				VecX tailTMP = bM.tail(ntail);
				bM.segment(io, ntail) = tailTMP;
				bM.tail<8>() = bTmp;

				MatXX HtmpCol = HM.block(0, io, odim, 8);
				MatXX rightColsTmp = HM.rightCols(ntail);
				HM.block(0, io, odim, ntail) = rightColsTmp;
				HM.rightCols(8) = HtmpCol;

				MatXX HtmpRow = HM.block(io, 0, 8, odim);
				MatXX botRowsTmp = HM.bottomRows(ntail);
				HM.block(io, 0, ntail, odim) = botRowsTmp;
				HM.bottomRows(8) = HtmpRow;
			}


			// marginalize. First add prior here, instead of to active.
			HM.bottomRightCorner<8, 8>().diagonal() += fh->prior;
			bM.tail<8>() += fh->prior.cwiseProduct(fh->delta_prior);

			VecX SVec = (HM.diagonal().cwiseAbs() + VecX::Constant(HM.cols(), 10)).cwiseSqrt();
			VecX SVecI = SVec.cwiseInverse();

			// scale!
			MatXX HMScaled = SVecI.asDiagonal() * HM * SVecI.asDiagonal();
			VecX bMScaled = SVecI.asDiagonal() * bM;

			// invert bottom part!
			Mat88 hpi = HMScaled.bottomRightCorner<8, 8>();
			/*hpi = 0.5f * (hpi + hpi);
			hpi = hpi.inverse();
			hpi = 0.5f * (hpi + hpi);*/
			hpi = util::MatrixInverter::invertPosDef(hpi);

			// schur-complement!
			MatXX bli = HMScaled.bottomLeftCorner(8, ndim).transpose() * hpi.selfadjointView<Eigen::Upper>().toDenseMatrix();
			HMScaled.topLeftCorner(ndim, ndim).noalias() -= bli * HMScaled.bottomLeftCorner(8, ndim);
			bMScaled.head(ndim).noalias() -= bli * bMScaled.tail<8>();

			// unscale!
			HMScaled = SVec.asDiagonal() * HMScaled * SVec.asDiagonal();
			bMScaled = SVec.asDiagonal() * bMScaled;

			// set.
			HM = 0.5 * (HMScaled.topLeftCorner(ndim, ndim) + HMScaled.topLeftCorner(ndim, ndim).transpose());
			bM = bMScaled.head(ndim);

			// remove from vector, without changing the order!
			for (unsigned int i = fh->idx; i + 1 < frames.size(); i++) {
				frames[i] = frames[i + 1];
				frames[i]->idx = i;
			}
			frames.pop_back();
			nFrames--;

			assert((int)frames.size() * 8 + CPARS == (int)HM.rows());
			assert((int)frames.size() * 8 + CPARS == (int)HM.cols());
			assert((int)frames.size() * 8 + CPARS == (int)bM.size());
			assert((int)frames.size() == (int)nFrames);

			EFIndicesValid = false;
			EFAdjointsValid = false;
			EFDeltaValid = false;

			makeIDX();
		}

		void EnergyFunctional::removePoint(shared_ptr<PointHessian> ph) {
			for (auto &r : ph->residuals) {
				connectivityMap[(((uint64_t)r->host.lock()->frameID) << 32) +
					((uint64_t)r->target.lock()->frameID)][0]--;
				nResiduals--;
			}
			ph->residuals.clear();
			if (!ph->alreadyRemoved)
				nPoints--;
			EFIndicesValid = false;
		}

		void EnergyFunctional::marginalizePointsF() {

			allPointsToMarg.clear();

			// go through all points to see which to marg
			for (auto f : frames) {
				for (shared_ptr<Feature> feat : f->frame->features) {

					if (feat->status == Feature::FeatureStatus::VALID &&
						feat->point->status == Point::PointStatus::MARGINALIZED) {
						shared_ptr<PointHessian> p = feat->point->mpPH;
						p->priorF *= setting_idepthFixPriorMargFac;
						for (auto r : p->residuals)
							if (r->isActive())
								connectivityMap[(((uint64_t)r->host.lock()->frameID) << 32) +
								((uint64_t)r->target.lock()->frameID)][1]++;
						allPointsToMarg.push_back(p);
					}
				}
			}

			accSSE_bot->setZero(nFrames);
			accSSE_top_A->setZero(nFrames);

			for (auto p : allPointsToMarg) {
				accSSE_top_A->addPoint<2>(p, this);
				accSSE_bot->addPoint(p, false);
				removePoint(p);
			}

			MatXX M, Msc;
			VecX Mb, Mbsc;
			accSSE_top_A->stitchDouble(M, Mb, this, false, false);
			accSSE_bot->stitchDouble(Msc, Mbsc, this);

			resInM += accSSE_top_A->nres[0];

			MatXX H = M - Msc;
			VecX b = Mb - Mbsc;

			if (setting_solverMode & SOLVER_ORTHOGONALIZE_POINTMARG) {
				bool haveFirstFrame = false;
				for (auto f : frames)
					if (f->frameID == 0)
						haveFirstFrame = true;
				if (!haveFirstFrame)
					orthogonalize(&bM, &HM);
			}

			HM += setting_margWeightFac * H;
			bM += setting_margWeightFac * b;

			if (setting_solverMode & SOLVER_ORTHOGONALIZE_FULL)
				orthogonalize(&bM, &HM);

			EFIndicesValid = false;
			makeIDX();
		}

		void EnergyFunctional::dropPointsF() {

			for (auto f : frames) {
				for (shared_ptr<Feature> feat : f->frame->features) {
					if (feat->point &&
						(feat->point->status == Point::PointStatus::OUTLIER ||
							feat->point->status == Point::PointStatus::OUT)
						&& feat->point->mpPH->alreadyRemoved == false) {
						removePoint(feat->point->mpPH);
					}
				}
			}
			EFIndicesValid = false;
			makeIDX();
		}

		void EnergyFunctional::solveSystemF(int iteration, double lambda, shared_ptr<CalibHessian> HCalib, shared_ptr<inertial::InertialHessian> HInertial) {

			if (setting_solverMode & SOLVER_USE_GN) lambda = 0;
			if (setting_solverMode & SOLVER_FIX_LAMBDA) lambda = 1e-5;

			int scaleGravityOffset = 0;

			if (setting_vi_optimize_scale_and_gravity_direction)
				scaleGravityOffset = 4;

			assert(EFDeltaValid);
			assert(EFAdjointsValid);
			assert(EFIndicesValid);

			// construct matricies
			MatXX HL_top, HA_top, H_sc;
			VecX bL_top, bA_top, bM_top, b_sc;

			accumulateAF_MT(HA_top, bA_top, multiThreading);
			accumulateLF_MT(HL_top, bL_top, multiThreading);
			accumulateSCF_MT(H_sc, b_sc, multiThreading);


		/*	LOG(INFO) << "norm(HL_top): " << HL_top.selfadjointView<Eigen::Upper>().toDenseMatrix().norm();
			LOG(INFO) << "norm(HM): " << HM.selfadjointView<Eigen::Upper>().toDenseMatrix().norm();
			LOG(INFO) << "norm(HA_top): " << HA_top.selfadjointView<Eigen::Upper>().toDenseMatrix().norm();*/

			if (setting_vi_enable)
				combineInertialHessians(lambda);

			bM_top = (bM + HM * getStitchedDeltaF());

			MatXX HFinal_top;
			VecX bFinal_top;

			if (setting_vi_use_schur_complement || !setting_vi_enable)
			{
				HFinal_top = MatXX::Zero(HA_top.rows(), HA_top.cols());
				bFinal_top = VecX::Zero(bA_top.rows());
			}
			else {
				HFinal_top = MatXX::Zero(HA_top.rows() + nFrames * 15 + scaleGravityOffset, HA_top.cols() + nFrames * 15 + scaleGravityOffset);
				bFinal_top = VecX::Zero(bA_top.rows() + nFrames * 15 + scaleGravityOffset);
			}

			if (setting_solverMode & SOLVER_ORTHOGONALIZE_SYSTEM) {
				// have a look if prior is there.
				bool haveFirstFrame = false;
				for (auto f : frames)
					if (f->frameID == 0)
						haveFirstFrame = true;

				MatXX HT_act = HL_top + HA_top - H_sc;
				VecX bT_act = bL_top + bA_top - b_sc;

				if (!haveFirstFrame)
					orthogonalize(&bT_act, &HT_act);

				HFinal_top = HT_act + HM;
				bFinal_top = bT_act + bM_top;
				//lastHS = HFinal_top;
				//lastbS = bFinal_top;

				for (int i = 0; i < 8 * nFrames + CPARS; i++)
					HFinal_top(i, i) *= (1 + lambda);
			}
			else {
				HFinal_top.topLeftCorner(8 * nFrames + CPARS, 8 * nFrames + CPARS).triangularView<Eigen::Upper>() = HL_top + HM + HA_top;
				bFinal_top.head(8 * nFrames + CPARS) = bL_top + bM_top + bA_top - b_sc / (1 + lambda);

				//lastHS = HFinal_top - H_sc - H_I_sc.selfadjointView<Eigen::Upper>().toDenseMatrix();

				if (setting_vi_enable) {
					HFinal_top.topLeftCorner(8 * nFrames + CPARS, 8 * nFrames + CPARS).triangularView<Eigen::Upper>() += H_I.selfadjointView<Eigen::Upper>().toDenseMatrix();
					bFinal_top.head(8 * nFrames + CPARS) += b_I;

					if (setting_vi_use_schur_complement) {

						bFinal_top -= b_I_sc;
					}
					else {
						bFinal_top.tail(nFrames * 15 + scaleGravityOffset) += bb_I;
						HFinal_top.bottomRightCorner(nFrames * 15 + scaleGravityOffset, nFrames * 15 + scaleGravityOffset).triangularView<Eigen::Upper>() += Hbb_I;
						HFinal_top.topRightCorner(CPARS + 8 * nFrames, scaleGravityOffset + 15 * nFrames) += Hab_I;
					}
					//lastHS -= H_I_sc.selfadjointView<Eigen::Upper>();
				}


				//lastbS = bFinal_top;

				for (int i = 0; i < 8 * nFrames + CPARS; i++)
					HFinal_top(i, i) *= (1 + lambda);

				if (setting_vi_enable)
					if (setting_vi_use_schur_complement)
						HFinal_top.triangularView<Eigen::Upper>() -= H_I_sc.selfadjointView<Eigen::Upper>().toDenseMatrix();

				HFinal_top.topLeftCorner(8 * nFrames + CPARS, 8 * nFrames + CPARS).triangularView<Eigen::Upper>() -= (H_sc) * (1.0f / (1 + lambda));
			}

			// get the result
			VecX x;
			if (setting_solverMode & SOLVER_SVD) {
				VecX SVecI = HFinal_top.diagonal().cwiseSqrt().cwiseInverse();
				MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top.selfadjointView<Eigen::Upper>().toDenseMatrix() * SVecI.asDiagonal();
				VecX bFinalScaled = SVecI.asDiagonal() * bFinal_top;
				Eigen::JacobiSVD<MatXX> svd(HFinalScaled, Eigen::ComputeThinU | Eigen::ComputeThinV);

				VecX S = svd.singularValues();
				double minSv = 1e10, maxSv = 0;
				for (int i = 0; i < S.size(); i++) {
					if (S[i] < minSv) minSv = S[i];
					if (S[i] > maxSv) maxSv = S[i];
				}

				VecX Ub = svd.matrixU().transpose() * bFinalScaled;
				int setZero = 0;
				for (int i = 0; i < Ub.size(); i++) {
					if (S[i] < setting_solverModeDelta * maxSv) {
						Ub[i] = 0;
						setZero++;
					}

					if ((setting_solverMode & SOLVER_SVD_CUT7) && (i >= Ub.size() - 7)) {
						Ub[i] = 0;
						setZero++;
					}
					else Ub[i] /= S[i];
				}
				x = SVecI.asDiagonal() * svd.matrixV() * Ub;

			}
			else {

				VecX SVecI = (HFinal_top.diagonal() + VecX::Constant(HFinal_top.cols(), 10)).cwiseSqrt().cwiseInverse();
				MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top.selfadjointView<Eigen::Upper>().toDenseMatrix() * SVecI.asDiagonal();

				/*
				x = SVecI.asDiagonal() * HFinalScaled.ldlt().solve(
						SVecI.asDiagonal() * bFinal_top);//  SVec.asDiagonal() * svd.matrixV() * Ub;
						*/

				x = SVecI.asDiagonal() * HFinalScaled.selfadjointView<Eigen::Upper>().ldlt().solve(
					SVecI.asDiagonal() * bFinal_top);//  SVec.asDiagonal() * svd.matrixV() * Ub;

			}

			if ((setting_solverMode & SOLVER_ORTHOGONALIZE_X) ||
				(iteration >= 2 && (setting_solverMode & SOLVER_ORTHOGONALIZE_X_LATER))) {
				VecX xOld = x;
				orthogonalize(&x, 0);
			}

			//if (!std::isfinite(x.squaredNorm()))
			//{
			//	//std::cout << "H:" << std::endl << HFinal_top << std::endl;
			//	std::cout << "HFinal_top:" << std::endl << HFinal_top.selfadjointView<Eigen::Upper>().toDenseMatrix().eigenvalues().transpose().format(setting_vi_format) << std::endl << std::endl;
			//	std::cout << "H_I:" << std::endl << H_I.eigenvalues().transpose().format(setting_vi_format) << std::endl << std::endl;
			//	//std::cout << "Hbb_I_inv:" << std::endl << Hbb_I_inv.selfadjointView<Eigen::Upper>().toDenseMatrix().eigenvalues().transpose().format(setting_vi_format) << std::endl << std::endl;
			//	std::cout << "HM_I:" << std::endl << HM_I.selfadjointView<Eigen::Upper>().toDenseMatrix().eigenvalues().transpose().format(setting_vi_format) << std::endl << std::endl;
			//	std::cout << "(HL_top + HM + HA_top - H_sc):" << std::endl << (HL_top + HM + HA_top - H_sc).eigenvalues().transpose().format(setting_vi_format) << std::endl << std::endl;
			//	if (setting_vi_use_schur_complement)
			//		std::cout << "(H_I - H_I_sc):" << std::endl << (H_I.selfadjointView<Eigen::Upper>().toDenseMatrix() - H_I_sc.selfadjointView<Eigen::Upper>().toDenseMatrix()).eigenvalues().transpose().format(setting_vi_format) << std::endl << std::endl;
			//	//std::cout << "Hbb_I_inv:" << std::endl << Hbb_I_inv.selfadjointView<Eigen::Upper>().toDenseMatrix() << std::endl << std::endl;
			//	//std::cout << "b:" << std::endl << bFinal_top << std::endl;
			//}

			/*LOG(INFO) << "norm(HFinal_top): " << HFinal_top.selfadjointView<Eigen::Upper>().toDenseMatrix().norm();
			LOG(INFO) << "norm(H_I): " << H_I.selfadjointView<Eigen::Upper>().toDenseMatrix().norm();

			LOG(INFO) << "norm(Hab_I): " << Hab_I.selfadjointView<Eigen::Upper>().toDenseMatrix().norm();
			LOG(INFO) << "norm(Hbb_I): " << Hbb_I.selfadjointView<Eigen::Upper>().toDenseMatrix().norm();

			LOG(INFO) << "norm(HL_top): " << HL_top.selfadjointView<Eigen::Upper>().toDenseMatrix().norm();
			LOG(INFO) << "norm(HM): " << HM.selfadjointView<Eigen::Upper>().toDenseMatrix().norm();
			LOG(INFO) << "norm(HA_top): " << HA_top.selfadjointView<Eigen::Upper>().toDenseMatrix().norm();

			LOG(INFO) << "norm(bFinal_top): " << bFinal_top.norm();
			LOG(INFO) << "norm(b_I): " << b_I.norm();
			LOG(INFO) << "norm(x): " << x.norm();*/

			lastX = x.head(8 * nFrames + CPARS);
			currentLambda = lambda;

			if (setting_vi_enable)
				resubstituteInertial(x, HInertial);

			resubstituteF_MT(lastX, HCalib, multiThreading);
			currentLambda = 0;

		}

		double EnergyFunctional::calcMEnergyF() {
			assert(EFDeltaValid);
			assert(EFAdjointsValid);
			assert(EFIndicesValid);
			VecX delta = getStitchedDeltaF();
			return delta.dot(2 * bM + HM * delta);
		}

		double EnergyFunctional::calcLEnergyF_MT() {
			assert(EFDeltaValid);
			assert(EFAdjointsValid);
			assert(EFIndicesValid);

			double E = 0;
			for (auto f : frames)
				E += f->delta_prior.cwiseProduct(f->prior).dot(f->delta_prior);

			E += cDeltaF.cwiseProduct(cPriorF).dot(cDeltaF);

			red->reduce(bind(&EnergyFunctional::calcLEnergyPt,
				this, _1, _2, _3, _4), 0, allPoints.size(), 50);

			// E += calcLEnergyFeat(); // calc feature's energy

			return E + red->stats[0];
		}

		void EnergyFunctional::makeIDX() {

			for (unsigned int idx = 0; idx < frames.size(); idx++)
				frames[idx]->idx = idx;

			allPoints.clear();

			for (auto f : frames) {
				for (shared_ptr<Feature> feat : f->frame->features) {
					if (feat->status == Feature::FeatureStatus::VALID &&
						feat->point->status == Point::PointStatus::ACTIVE) {
						shared_ptr<PointHessian> p = feat->point->mpPH;
						allPoints.push_back(p);
						for (auto &r : p->residuals) {
							r->hostIDX = r->host.lock()->idx;
							r->targetIDX = r->target.lock()->idx;
						}
					}
				}
			}
			EFIndicesValid = true;
		}

		void EnergyFunctional::setDeltaF(shared_ptr<CalibHessian> HCalib) {
			if (adHTdeltaF != 0) delete[] adHTdeltaF;
			adHTdeltaF = new Mat18f[nFrames * nFrames];
			for (int h = 0; h < nFrames; h++)
				for (int t = 0; t < nFrames; t++) {
					int idx = h + t * nFrames;
					adHTdeltaF[idx] =
						frames[h]->get_state_minus_stateZero().head<8>().cast<float>().transpose() * adHostF[idx]
						+
						frames[t]->get_state_minus_stateZero().head<8>().cast<float>().transpose() * adTargetF[idx];
				}

			cDeltaF = HCalib->value_minus_value_zero.cast<float>();
			for (auto f : frames) {
				f->delta = f->get_state_minus_stateZero().head<8>();
				f->delta_prior = (f->get_state_minus_statePriorZero()).head<8>();

				for (auto feat : f->frame->features) {
					if (feat->status == Feature::FeatureStatus::VALID && feat->point &&
						feat->point->status == Point::PointStatus::ACTIVE) {
						auto p = feat->point->mpPH;
						p->deltaF = p->idepth - p->idepth_zero;
					}
				}
			}
			EFDeltaValid = true;
		}

		void EnergyFunctional::setAdjointsF(shared_ptr<CalibHessian> Hcalib) {

			if (adHost != 0) delete[] adHost;
			if (adTarget != 0) delete[] adTarget;

			adHost = new Mat88[nFrames * nFrames];
			adTarget = new Mat88[nFrames * nFrames];

			for (int h = 0; h < nFrames; h++)
				for (int t = 0; t < nFrames; t++) {
					shared_ptr<FrameHessian> host = frames[h];
					shared_ptr<FrameHessian> target = frames[t];

					SE3 hostToTarget = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();

					Mat88 AH = Mat88::Identity();
					Mat88 AT = Mat88::Identity();

					AH.topLeftCorner<6, 6>() = -hostToTarget.Adj().transpose();
					AT.topLeftCorner<6, 6>() = Mat66::Identity();


					Vec2f affLL = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l_0(),
						target->aff_g2l_0()).cast<float>();
					AT(6, 6) = -affLL[0];
					AH(6, 6) = affLL[0];
					AT(7, 7) = -1;
					AH(7, 7) = affLL[0];

					AH.block<3, 8>(0, 0) *= SCALE_XI_TRANS;
					AH.block<3, 8>(3, 0) *= SCALE_XI_ROT;
					AH.block<1, 8>(6, 0) *= SCALE_A;
					AH.block<1, 8>(7, 0) *= SCALE_B;
					AT.block<3, 8>(0, 0) *= SCALE_XI_TRANS;
					AT.block<3, 8>(3, 0) *= SCALE_XI_ROT;
					AT.block<1, 8>(6, 0) *= SCALE_A;
					AT.block<1, 8>(7, 0) *= SCALE_B;

					adHost[h + t * nFrames] = AH;
					adTarget[h + t * nFrames] = AT;
				}

			cPrior = VecC::Constant(setting_initialCalibHessian);


			if (adHostF != 0) delete[] adHostF;
			if (adTargetF != 0) delete[] adTargetF;
			adHostF = new Mat88f[nFrames * nFrames];
			adTargetF = new Mat88f[nFrames * nFrames];

			for (int h = 0; h < nFrames; h++)
				for (int t = 0; t < nFrames; t++) {
					adHostF[h + t * nFrames] = adHost[h + t * nFrames].cast<float>();
					adTargetF[h + t * nFrames] = adTarget[h + t * nFrames].cast<float>();
				}

			cPriorF = cPrior.cast<float>();
			EFAdjointsValid = true;
		}

		void EnergyFunctional::resubstituteF_MT(const VecX &x, shared_ptr<CalibHessian> HCalib, bool MT) {
			assert(x.size() == CPARS + nFrames * 8);

			VecXf xF = x.cast<float>();
			HCalib->step = -x.head<CPARS>();

			Mat18f *xAd = new Mat18f[nFrames * nFrames];
			VecCf cstep = xF.head<CPARS>();
			for (auto h : frames) {
				h->step.head<8>() = -x.segment<8>(CPARS + 8 * h->idx);
				h->step.tail<2>().setZero();

				for (auto t : frames)
					xAd[nFrames * h->idx + t->idx] =
					xF.segment<8>(CPARS + 8 * h->idx).transpose() * adHostF[h->idx + nFrames * t->idx]
					+ xF.segment<8>(CPARS + 8 * t->idx).transpose() * adTargetF[h->idx + nFrames * t->idx];
			}

			if (MT)
				red->reduce(bind(&EnergyFunctional::resubstituteFPt,
					this, cstep, xAd, _1, _2, _3, _4), 0, allPoints.size(), 50);
			else
				resubstituteFPt(cstep, xAd, 0, allPoints.size(), 0, 0);

			delete[] xAd;
		}

		void EnergyFunctional::resubstituteFPt(const VecCf &xc, Mat18f *xAd, int min, int max, Vec10 *stats, int tid) {

			for (int k = min; k < max; k++) {
				auto p = allPoints[k];

				int ngoodres = 0;
				for (auto r : p->residuals)
					if (r->isActive())
						ngoodres++;

				if (ngoodres == 0) {
					p->step = 0;
					continue;
				}

				float b = p->bdSumF;
				b -= xc.dot(p->Hcd_accAF + p->Hcd_accLF);

				for (auto r : p->residuals) {
					if (!r->isActive()) continue;
					b -= xAd[r->hostIDX * nFrames + r->targetIDX] * r->JpJdF;
				}

				if (!std::isfinite(b) || std::isnan(b)) {
					return;
				}

				p->step = -b * p->HdiF / (1 + currentLambda);
			}
		}

		// accumulates & shifts L.
		void EnergyFunctional::accumulateAF_MT(MatXX &H, VecX &b, bool MT) {
			if (MT) {
				red->reduce(bind(&AccumulatedTopHessianSSE::setZero, accSSE_top_A, nFrames, _1, _2, _3, _4), 0,
					0, 0);
				red->reduce(bind(&AccumulatedTopHessianSSE::addPointsInternal<0>,
					accSSE_top_A, &allPoints, this, _1, _2, _3, _4), 0, allPoints.size(), 50);
				accSSE_top_A->stitchDoubleMT(red, H, b, this, false, true);
				resInA = accSSE_top_A->nres[0];
			}
			else {
				accSSE_top_A->setZero(nFrames);
				int cntPointAdded = 0;
				for (auto f : frames) {
					for (shared_ptr<Feature> &feat : f->frame->features) {
						if (feat->status == Feature::FeatureStatus::VALID && feat->point &&
							feat->point->status == Point::PointStatus::ACTIVE) {
							auto p = feat->point->mpPH;
							accSSE_top_A->addPoint<0>(p, this);
							cntPointAdded++;
						}
					}
				}
				accSSE_top_A->stitchDoubleMT(red, H, b, this, false, false);
				resInA = accSSE_top_A->nres[0];
			}
		}

		// accumulates & shifts L.
		void EnergyFunctional::accumulateLF_MT(MatXX &H, VecX &b, bool MT) {
			if (MT) {
				red->reduce(bind(&AccumulatedTopHessianSSE::setZero, accSSE_top_L, nFrames, _1, _2, _3, _4), 0,
					0, 0);
				red->reduce(bind(&AccumulatedTopHessianSSE::addPointsInternal<1>,
					accSSE_top_L, &allPoints, this, _1, _2, _3, _4), 0, allPoints.size(), 50);
				accSSE_top_L->stitchDoubleMT(red, H, b, this, true, true);
				resInL = accSSE_top_L->nres[0];
			}
			else {
				accSSE_top_L->setZero(nFrames);
				int cntPointAdded = 0;
				for (auto f : frames) {
					for (auto feat : f->frame->features) {
						if (feat->status == Feature::FeatureStatus::VALID &&
							feat->point->status == Point::PointStatus::ACTIVE) {
							auto p = feat->point->mpPH;
							accSSE_top_L->addPoint<1>(p, this);
							cntPointAdded++;
						}
					}
				}
				//LOG(INFO) << "HL points: " << cntPointAdded;
				accSSE_top_L->stitchDoubleMT(red, H, b, this, true, false);
				resInL = accSSE_top_L->nres[0];
			}
		}

		void EnergyFunctional::accumulateSCF_MT(MatXX &H, VecX &b, bool MT) {
			if (MT) {
				red->reduce(bind(&AccumulatedSCHessianSSE::setZero, accSSE_bot, nFrames, _1, _2, _3, _4), 0, 0,
					0);
				red->reduce(bind(&AccumulatedSCHessianSSE::addPointsInternal,
					accSSE_bot, &allPoints, true, _1, _2, _3, _4), 0, allPoints.size(), 50);
				accSSE_bot->stitchDoubleMT(red, H, b, this, true);
			}
			else {
				accSSE_bot->setZero(nFrames);
				int cntPointAdded = 0;
				for (auto f : frames) {
					for (auto feat : f->frame->features) {
						if (feat->status == Feature::FeatureStatus::VALID &&
							feat->point->status == Point::PointStatus::ACTIVE) {
							auto p = feat->point->mpPH;
							accSSE_bot->addPoint(p, true);
							cntPointAdded++;
						}
					}
				}
				accSSE_bot->stitchDoubleMT(red, H, b, this, false);
			}
		}

		void EnergyFunctional::calcLEnergyPt(int min, int max, Vec10 *stats, int tid) {

			Accumulator11 E;
			E.initialize();
			VecCf dc = cDeltaF;

			for (int i = min; i < max; i++) {
				auto p = allPoints[i];
				float dd = p->deltaF;

				for (auto r : p->residuals) {
					if (!r->isLinearized || !r->isActive()) continue;

					Mat18f dp = adHTdeltaF[r->hostIDX + nFrames * r->targetIDX];
					shared_ptr<RawResidualJacobian> rJ = r->J;

					// compute Jp*delta
					float Jp_delta_x_1 = rJ->Jpdxi[0].dot(dp.head<6>())
						+ rJ->Jpdc[0].dot(dc)
						+ rJ->Jpdd[0] * dd;

					float Jp_delta_y_1 = rJ->Jpdxi[1].dot(dp.head<6>())
						+ rJ->Jpdc[1].dot(dc)
						+ rJ->Jpdd[1] * dd;

					__m128 Jp_delta_x = _mm_set1_ps(Jp_delta_x_1);
					__m128 Jp_delta_y = _mm_set1_ps(Jp_delta_y_1);
					__m128 delta_a = _mm_set1_ps((float)(dp[6]));
					__m128 delta_b = _mm_set1_ps((float)(dp[7]));

					for (int i = 0; i + 3 < patternNum; i += 4) {
						// PATTERN: E = (2*res_toZeroF + J*delta) * J*delta.
						__m128 Jdelta = _mm_mul_ps(_mm_load_ps(((float *)(rJ->JIdx)) + i), Jp_delta_x);
						Jdelta = _mm_add_ps(Jdelta,
							_mm_mul_ps(_mm_load_ps(((float *)(rJ->JIdx + 1)) + i), Jp_delta_y));
						Jdelta = _mm_add_ps(Jdelta, _mm_mul_ps(_mm_load_ps(((float *)(rJ->JabF)) + i), delta_a));
						Jdelta = _mm_add_ps(Jdelta, _mm_mul_ps(_mm_load_ps(((float *)(rJ->JabF + 1)) + i), delta_b));

						__m128 r0 = _mm_load_ps(((float *)&r->res_toZeroF) + i);
						r0 = _mm_add_ps(r0, r0);
						r0 = _mm_add_ps(r0, Jdelta);
						Jdelta = _mm_mul_ps(Jdelta, r0);
						E.updateSSENoShift(Jdelta);
					}
					for (int i = ((patternNum >> 2) << 2); i < patternNum; i++) {
						float Jdelta = rJ->JIdx[0][i] * Jp_delta_x_1 + rJ->JIdx[1][i] * Jp_delta_y_1 +
							rJ->JabF[0][i] * dp[6] + rJ->JabF[1][i] * dp[7];
						E.updateSingleNoShift((float)(Jdelta * (Jdelta + 2 * r->res_toZeroF[i])));
					}
				}

				E.updateSingle(p->deltaF * p->deltaF * p->priorF);
			}
			E.finish();
			(*stats)[0] += E.A;
		}


		void EnergyFunctional::orthogonalize(VecX *b, MatXX *H) {

			std::vector<VecX> ns;
			ns.insert(ns.end(), lastNullspaces_pose.begin(), lastNullspaces_pose.end());
			ns.insert(ns.end(), lastNullspaces_scale.begin(), lastNullspaces_scale.end());

			// make Nullspaces matrix
			MatXX N(ns[0].rows(), ns.size());
			for (unsigned int i = 0; i < ns.size(); i++)
				N.col(i) = ns[i].normalized();

			// compute Npi := N * (N' * N)^-1 = pseudo inverse of N.
			Eigen::JacobiSVD<MatXX> svdNN(N, Eigen::ComputeThinU | Eigen::ComputeThinV);

			VecX SNN = svdNN.singularValues();
			double minSv = 1e10, maxSv = 0;
			for (int i = 0; i < SNN.size(); i++) {
				if (SNN[i] < minSv) minSv = SNN[i];
				if (SNN[i] > maxSv) maxSv = SNN[i];
			}
			for (int i = 0; i < SNN.size(); i++) {
				if (SNN[i] > setting_solverModeDelta * maxSv)
					SNN[i] = 1.0 / SNN[i];
				else SNN[i] = 0;
			}

			MatXX Npi = svdNN.matrixU() * SNN.asDiagonal() * svdNN.matrixV().transpose();    // [dim] x 9.
			MatXX NNpiT = N * Npi.transpose();    // [dim] x [dim].
			MatXX NNpiTS = 0.5 * (NNpiT + NNpiT.transpose());    // = N * (N' * N)^-1 * N'.

			if (b != 0) *b -= NNpiTS * *b;
			if (H != 0) *H -= NNpiTS * *H * NNpiTS;
		}

		void EnergyFunctional::combineInertialHessians(double lambda)
		{
			int scaleGravityOffset = 0;

			if (setting_vi_optimize_scale_and_gravity_direction)
				scaleGravityOffset = 4;

			Hab_I = MatXX::Zero(CPARS + 8 * nFrames, scaleGravityOffset + 15 * nFrames);
			Hbb_I = MatXX::Zero(scaleGravityOffset + 15 * nFrames, scaleGravityOffset + 15 * nFrames);
			bb_I = VecX::Zero(scaleGravityOffset + 15 * nFrames);
			H_I = MatXX::Zero(CPARS + 8 * nFrames, CPARS + 8 * nFrames);
			b_I = VecX::Zero(CPARS + 8 * nFrames);
			H_I_sc = MatXX::Zero(CPARS + 8 * nFrames, CPARS + 8 * nFrames);

			VecX deltaX = VecX::Zero(scaleGravityOffset + 15 * nFrames);

			int index = 0;

			Mat1515 S2_inv = S.block<15, 15>(10, 10).inverse();

			for (auto f : frames)
			{
				Mat2525 H = S * f->inertialFrameHessian->H * S;
				Vec25 b = S * f->inertialFrameHessian->b;

				H_I.block<6, 6>(CPARS + 8 * index, CPARS + 8 * index).triangularView<Eigen::Upper>() += H.block<6, 6>(0, 0);
				b_I.block<6, 1>(CPARS + 8 * index, 0) -= b.block<6, 1>(0, 0);

				if (setting_vi_optimize_scale_and_gravity_direction) {
					Hbb_I.block<4, 4>(0, 0).triangularView<Eigen::Upper>() += H.block<4, 4>(6, 6);
					Hbb_I.block<4, 15>(0, 4 + 15 * index) += H.block<4, 15>(6, 10);
				}

				Hbb_I.block<15, 15>(scaleGravityOffset + 15 * index, scaleGravityOffset + 15 * index).triangularView<Eigen::Upper>() += H.block<15, 15>(10, 10);

				//Hbb_I.block<15, 4>(4 + 15 * index, 0) += H.block<15, 4>(10, 6);

				if (f->inertialFrameHessian->from != nullptr)
				{
					Mat1515 Hab = S.block<15, 15>(10, 10) * f->inertialFrameHessian->from->H_from_to * S.block<15, 15>(10, 10);
					Hbb_I.block<15, 15>(scaleGravityOffset + 15 * index, scaleGravityOffset + 15 * (index + 1)) += Hab;
					//Hbb_I.block<15, 15>(4 + 15 * (index + 1), 4 + 15 * index) += Hab.transpose();
				}

				if (setting_vi_optimize_scale_and_gravity_direction)
					bb_I.block<4, 1>(0, 0) -= b.block<4, 1>(6, 0);
				bb_I.block<15, 1>(scaleGravityOffset + 15 * index, 0) -= b.block<15, 1>(10, 0);

				if (setting_vi_optimize_scale_and_gravity_direction)
					Hab_I.block<6, 4>(CPARS + 8 * index, 0) += H.block<6, 4>(0, 6);
				Hab_I.block<6, 15>(CPARS + 8 * index, scaleGravityOffset + 15 * index) += H.block<6, 15>(0, 10);

				deltaX.segment<15>(scaleGravityOffset + index * 15) = S2_inv * f->inertialFrameHessian->x;
				index++;
			}

			//if (Hbb_I.selfadjointView<Eigen::Upper>().eigenvalues().real().minCoeff() < 0.0)
			//	LOG(INFO) << Hbb_I.eigenvalues().transpose().format(setting_vi_format);

			//if (HM_I.selfadjointView<Eigen::Upper>().eigenvalues().real().minCoeff() < 0.0)
			//	LOG(INFO) << Hbb_I.eigenvalues().transpose().format(setting_vi_format);

			//if ((HM_I+ Hbb_I).selfadjointView<Eigen::Upper>().eigenvalues().real().minCoeff() < 0.0)
			//	LOG(INFO) << (HM_I + Hbb_I).eigenvalues().transpose().format(setting_vi_format);

			Hbb_I.triangularView<Eigen::Upper>() += HM_I.selfadjointView<Eigen::Upper>().toDenseMatrix();
			bb_I += bM_I + HM_I.selfadjointView<Eigen::Upper>() * deltaX;

			for (int i = 0; i < scaleGravityOffset + 15 * nFrames; i++)
				Hbb_I(i, i) *= (1 + lambda);

			if (setting_vi_use_schur_complement)
			{
				Hbb_I_inv = util::MatrixInverter::invertPosDef(Hbb_I, setting_use_fast_matrix_inverter);

				MatXX HabHbbinv;
				HabHbbinv = Hab_I * Hbb_I_inv.selfadjointView<Eigen::Upper>().toDenseMatrix();

				H_I_sc.triangularView<Eigen::Upper>() = HabHbbinv * Hab_I.transpose();
				b_I_sc = HabHbbinv * bb_I;
			}
		}

		void EnergyFunctional::marginalizeInertialFrameHessian(shared_ptr<FrameHessian> fh)
		{
			int scaleGravityOffset = 0;

			if (setting_vi_optimize_scale_and_gravity_direction)
				scaleGravityOffset = 4;

			Mat1515 S2 = S.block<15, 15>(10, 10);
			int ndim = (nFrames - 1) * 15 + scaleGravityOffset;// new dimension
			int odim = nFrames * 15 + scaleGravityOffset;// old dimension

			HM_I = HM_I.selfadjointView<Eigen::Upper>().toDenseMatrix();

			//LOG(INFO) << "HM_I(1):" << (HM_I).eigenvalues().transpose().format(setting_vi_format);

			if ((int)fh->idx != (int)frames.size() - 1) {
				int io = fh->idx * 15 + scaleGravityOffset;    // index of frame to move to end
				int ntail = 15 * (nFrames - fh->idx - 1);
				assert((io + 15 + ntail) == nFrames * 15 + scaleGravityOffset);

				Vec15 bTmp = bM_I.segment<15>(io);
				VecX tailTMP = bM_I.tail(ntail);
				bM_I.segment(io, ntail) = tailTMP;
				bM_I.tail<15>() = bTmp;

				MatXX HtmpCol = HM_I.block(0, io, odim, 15);
				MatXX rightColsTmp = HM_I.rightCols(ntail);
				HM_I.block(0, io, odim, ntail) = rightColsTmp;
				HM_I.rightCols(15) = HtmpCol;

				MatXX HtmpRow = HM_I.block(io, 0, 15, odim);
				MatXX botRowsTmp = HM_I.bottomRows(ntail);
				HM_I.block(io, 0, ntail, odim) = botRowsTmp;
				HM_I.bottomRows(15) = HtmpRow;
			}

			if (fh->inertialFrameHessian->from != nullptr)
			{
				HM_I.bottomRightCorner<15, 15>().triangularView<Eigen::Upper>() += setting_vi_marginalization_weight * S2 * fh->inertialFrameHessian->from->H_from.selfadjointView<Eigen::Upper>().toDenseMatrix() * S2;
				HM_I.block<15, 15>(fh->idx * 15 + scaleGravityOffset, fh->idx * 15 + scaleGravityOffset).triangularView<Eigen::Upper>() += setting_vi_marginalization_weight * S2 * fh->inertialFrameHessian->from->H_to.selfadjointView<Eigen::Upper>().toDenseMatrix() * S2;
				HM_I.block<15, 15>(fh->idx * 15 + scaleGravityOffset, ndim) += setting_vi_marginalization_weight * S2 * fh->inertialFrameHessian->from->H_from_to.transpose() * S2;

				bM_I.tail<15>() -= setting_vi_marginalization_weight * S2 * fh->inertialFrameHessian->from->b_from;
				bM_I.segment<15>(fh->idx * 15 + scaleGravityOffset) -= setting_vi_marginalization_weight * S2 * fh->inertialFrameHessian->from->b_to;
			}
			if (fh->inertialFrameHessian->to != nullptr)
			{
				HM_I.bottomRightCorner<15, 15>().triangularView<Eigen::Upper>() += setting_vi_marginalization_weight * S2 * fh->inertialFrameHessian->to->H_to.selfadjointView<Eigen::Upper>().toDenseMatrix() * S2;
				HM_I.block<15, 15>((fh->idx - 1) * 15 + scaleGravityOffset, (fh->idx - 1) * 15 + scaleGravityOffset).triangularView<Eigen::Upper>() += setting_vi_marginalization_weight * S2 * fh->inertialFrameHessian->to->H_from.selfadjointView<Eigen::Upper>().toDenseMatrix() * S2;
				HM_I.block<15, 15>((fh->idx - 1) * 15 + scaleGravityOffset, ndim) += setting_vi_marginalization_weight * S2 * fh->inertialFrameHessian->to->H_from_to * S2;

				bM_I.tail<15>() -= setting_vi_marginalization_weight * S2 * fh->inertialFrameHessian->to->b_to;
				bM_I.segment<15>((fh->idx - 1) * 15 + scaleGravityOffset) -= setting_vi_marginalization_weight * S2 * fh->inertialFrameHessian->to->b_from;
			}

			//LOG(INFO) << "HM_I(1.2):" << (HM_I.selfadjointView<Eigen::Upper>().toDenseMatrix()).eigenvalues().transpose().format(setting_vi_format);

			/*std::cout << "HM_I: " << std::endl << HM_I.bottomRightCorner<15, 15>().selfadjointView<Eigen::Upper>().toDenseMatrix() << std::endl;
			std::cout << "Hab: " << std::endl << HM_I.topRightCorner(ndim, 15) << std::endl;*/

			VecX SVec = (HM_I.diagonal().cwiseAbs() + VecX::Constant(HM_I.cols(), 10)).cwiseSqrt();
			VecX SVecI = SVec.cwiseInverse();

			// scale!
			MatXX HMScaled = MatXX::Zero(odim, odim);
			HMScaled.triangularView<Eigen::Upper>() = SVecI.asDiagonal() * HM_I.selfadjointView<Eigen::Upper>().toDenseMatrix() * SVecI.asDiagonal();
			VecX bMScaled = SVecI.asDiagonal() * bM_I;

			//LOG(INFO) << "HM_I(1.3.1):" << (HMScaled.selfadjointView<Eigen::Upper>().toDenseMatrix()).eigenvalues().transpose().format(setting_vi_format);
			//LOG(INFO) << "HM_I(1.3.2):" << (SVecI.asDiagonal().toDenseMatrix()).eigenvalues().transpose().format(setting_vi_format);

			// invert bottom part!
			MatXX hpi = Mat1515::Zero();
			hpi.triangularView<Eigen::Upper>() = HMScaled.bottomRightCorner<15, 15>().selfadjointView<Eigen::Upper>().toDenseMatrix();
			hpi = util::MatrixInverter::invertPosDef(hpi, setting_use_fast_matrix_inverter);

			//LOG(INFO) << "HM_I(1.4):" << (HMScaled.selfadjointView<Eigen::Upper>().toDenseMatrix()).eigenvalues().transpose().format(setting_vi_format);

			// schur-complement!
			MatXX bli = HMScaled.topRightCorner(ndim, 15) * hpi.selfadjointView<Eigen::Upper>();
			HMScaled.topLeftCorner(ndim, ndim).noalias() -= bli * HMScaled.topRightCorner(ndim, 15).transpose();
			bMScaled.head(ndim).noalias() -= bli * bMScaled.tail<15>();

			//LOG(INFO) << "HM_I(1.5):" << (HMScaled.selfadjointView<Eigen::Upper>().toDenseMatrix()).eigenvalues().transpose().format(setting_vi_format);

			// unscale!
			HMScaled = SVec.asDiagonal() * HMScaled.selfadjointView<Eigen::Upper>().toDenseMatrix() * SVec.asDiagonal();
			bMScaled = SVec.asDiagonal() * bMScaled;

			// set.
			HM_I = MatXX::Zero(ndim, ndim);
			HM_I.triangularView<Eigen::Upper>() = HMScaled.topLeftCorner(ndim, ndim);
			bM_I = bMScaled.head(ndim);

			//LOG(INFO) << "HM_I(2):" << (HM_I).eigenvalues().transpose().format(setting_vi_format);
		}

		void EnergyFunctional::resubstituteInertial(VecX x, shared_ptr<inertial::InertialHessian> HInertial)
		{
			int scaleGravityOffset = 0;

			if (setting_vi_optimize_scale_and_gravity_direction)
				scaleGravityOffset = 4;

			int index = 0;

			VecX xb;

			if (setting_vi_use_schur_complement)
			{
				xb = Hbb_I_inv.selfadjointView<Eigen::Upper>() * (bb_I - Hab_I.transpose() * x);
			}
			else
			{
				xb = x.tail(scaleGravityOffset + nFrames * 15);
			}

			if (setting_vi_optimize_scale_and_gravity_direction)
				HInertial->x_step = -S.block<4, 4>(6, 6) * xb.block<4, 1>(0, 0);
			else
				HInertial->x_step = Vec4::Zero();

			for (auto f : frames)
			{
				f->inertialFrameHessian->x_step = -S.block<15, 15>(10, 10) * xb.block<15, 1>(scaleGravityOffset + 15 * index, 0);
				index++;
			}
		}
	}
}