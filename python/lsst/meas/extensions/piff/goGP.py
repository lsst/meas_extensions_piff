import pickle
import numpy as np
import treegp
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import copy
from tqdm import tqdm

DX_TO_DU_UNITS = 0.2 * 10**3
XFoV_TO_U_UNITS = 100. * 0.2 / 3600.

__all__ =["goGPgo"]

class goGPgo():

    def __init__(
            self,
            pklIn='../data/visits_w23/2025052000330.pkl',
            repOut='../data/visits_w23_GP_OUT',
            repPlot='../plots',):
        
        self.dic = pickle.load(open(pklIn, 'rb'))
        self.repOut = repOut
        self.repPlot = repPlot

        xie, xib, logr = treegp.comp_eb_treecorr(self.dic['xFoV'] * XFoV_TO_U_UNITS,
                                                 self.dic['yFoV'] * XFoV_TO_U_UNITS,
                                                 self.dic['dx'] * DX_TO_DU_UNITS,
                                                 self.dic['dy'] * DX_TO_DU_UNITS,
                                                 rmin=20/3600, rmax=0.6, dlogr=0.3)
        MAXEB = max([np.max(xie), np.max(xib)])
        MINEB = min([np.min(xie), np.min(xib)])
        MAXEB += MAXEB * 0.1
        MINEB -= MINEB * 0.1

        self.MAXEB = MAXEB
        self.MINEB = MINEB

        self.point_size = 1
        self.RESDIUAL_LIM = np.nanstd(self.dic['dx']) * DX_TO_DU_UNITS
        self.CMAP = plt.cm.seismic
        

    def plot_visit(self, x, y, dx, dy, info="measured", namefig="visit.png"):


        plt.figure(figsize=(18,5))
        plt.subplots_adjust(wspace=0.3, right=0.99, left=0.05)


        plt.subplot(1, 3, 1)
        plt.scatter(x, y, c=dx, vmin=-self.RESDIUAL_LIM, vmax=self.RESDIUAL_LIM, cmap=self.CMAP, s=self.point_size)
        plt.subplot(1, 3, 2)
        plt.scatter(x, y, c=dy, vmin=-self.RESDIUAL_LIM, vmax=self.RESDIUAL_LIM, cmap=self.CMAP, s=self.point_size)

        plt.subplot(1, 3, 1)
        plt.axis('equal')
        cb = plt.colorbar()
        cb.set_label('dx (mas)')
        plt.xlabel('x (degree)')
        plt.ylabel('y (degree)')

        plt.subplot(1, 3, 2)
        plt.axis('equal')
        cb = plt.colorbar()
        cb.set_label('dy (mas)')
        plt.xlabel('x (degree)')

        xie, xib, logr = treegp.comp_eb_treecorr(x, y, dx, dy, rmin=20/3600, rmax=0.6, dlogr=0.3)

        plt.subplot(1, 3, 3)
        plt.scatter(np.exp(logr) * 60, xie, c='b', label='E-mode')
        plt.scatter(np.exp(logr) * 60, xib, c='r', label='B-mode')
        plt.xscale('log')
        xlim = plt.xlim()
        plt.plot(xlim, [0,0], 'k--')
        plt.xlim(xlim)
        plt.ylim(self.MINEB, self.MAXEB)
        plt.title(r'$\xi_{E/B}$ (mas$^2$)')
        plt.xlabel(r'$\Delta \theta$ (arcmin)')
        plt.legend()

        plt.suptitle(f'Visit: {self.dic["visit"]} | {info} | Band: {self.dic["band"]}')
        plt.savefig(os.path.join(self.repPlot, f"{namefig}"))
        plt.close()

        return xie, xib, logr
    
    def run_gp(self):

        kernel_init = kernel_init = "15**2 * AnisotropicVonKarman(invLam=array([[1./3000.**2,0],[0,1./3000.**2]]))"
        gpx = treegp.GPInterpolation(kernel=kernel_init,
                                optimizer='anisotropic',
                                normalize=True,
                                nbins=21,
                                min_sep=0.,
                                max_sep=0.3,
                                p0=[1, -0.2, -0.2])
        coord = np.array([self.dic['xFoV'] * XFoV_TO_U_UNITS, self.dic['yFoV'] * XFoV_TO_U_UNITS]).T
        rng = np.random.default_rng(1234)
        nPoints = len(coord)
        nTrain = 10000
        perm = rng.permutation(np.arange(nPoints))
        indice_train = perm[:nTrain]
        indice_test = perm[nTrain:]
        # Predict on dx:
        dx = self.dic['dx'][indice_train] * DX_TO_DU_UNITS
        dx_err = self.dic['sigpix'][indice_train] * DX_TO_DU_UNITS
        gpx.initialize(coord[indice_train], dx, y_err=dx_err)
        gpx.solve()
        # trick to avoid memory burnout at s3df
        # will interpolate CCD per CCD but using full focal
        # plane matrix
        detectors = list(set(self.dic['ccdId']))
        self.dx_predict = np.zeros_like(self.dic['dx'])
        for ccdId in tqdm(detectors):
            FilterDetector = (ccdId == self.dic['ccdId'])
            self.dx_predict[FilterDetector] = gpx.predict(coord[FilterDetector])
        # self.dx_predict = gpx.predict(coord)

        kernel_init = kernel_init = "15**2 * AnisotropicVonKarman(invLam=array([[1./3000.**2,0],[0,1./3000.**2]]))"
        gpy = treegp.GPInterpolation(kernel=kernel_init,
                                     optimizer='anisotropic',
                                     normalize=True,
                                     nbins=21,
                                     min_sep=0.,
                                     max_sep=0.2,
                                     p0=[1, -0.2, -0.2])

        # Predict on dy:
        dy = self.dic['dy'][indice_train] * DX_TO_DU_UNITS
        dy_err = self.dic['sigpix'][indice_train] * DX_TO_DU_UNITS
        gpy.initialize(coord[indice_train], dy, y_err=dy_err)
        gpy.solve()
        # trick to avoid memory burnout at s3df
        # will interpolate CCD per CCD but using full focal
        # plane matrix
        detectors = list(set(self.dic['ccdId']))
        self.dy_predict = np.zeros_like(self.dic['dx'])
        for ccdId in tqdm(detectors):
            FilterDetector = (ccdId == self.dic['ccdId'])
            self.dy_predict[FilterDetector] = gpy.predict(coord[FilterDetector])
        # self.dy_predict = gpy.predict(coord)


        self.gpx = gpx
        self.gpy = gpy

    def plotThemAll(self):

        self.xie, self.xib, self.logr = self.plot_visit(self.dic['xFoV'] * XFoV_TO_U_UNITS,
                                                        self.dic['yFoV'] * XFoV_TO_U_UNITS,
                                                        self.dic['dx'] * DX_TO_DU_UNITS,
                                                        self.dic['dy'] * DX_TO_DU_UNITS,
                                                        info="measured", namefig=f'{self.dic["visit"]}_1_measured.png')

        self.gpx.plot_fitted_kernel()
        plt.savefig(os.path.join(self.repPlot, f'{self.dic["visit"]}_dx_2pcf_fit.png'))
        plt.close()

        self.gpy.plot_fitted_kernel()
        plt.savefig(os.path.join(self.repPlot, f'{self.dic["visit"]}_dy_2pcf_fit.png'))
        plt.close()

        self.xieGP, self.xibGP, self.logr = self.plot_visit(self.dic['xFoV'] * XFoV_TO_U_UNITS,
                                                            self.dic['yFoV'] * XFoV_TO_U_UNITS,
                                                            self.dx_predict,
                                                            self.dy_predict,
                                                            info="GP model", namefig=f'{self.dic["visit"]}_0_gp.png')
        
        self.xieResiduals, self.xibResiduals, self.logr = self.plot_visit(self.dic['xFoV'] * XFoV_TO_U_UNITS,
                                                                          self.dic['yFoV'] * XFoV_TO_U_UNITS,
                                                                          self.dic['dx'] * DX_TO_DU_UNITS - self.dx_predict,
                                                                          self.dic['dy'] * DX_TO_DU_UNITS - self.dy_predict,
                                                                          info="residuals", namefig=f'{self.dic["visit"]}_2_residuals.png')

    def write_output(self):

        newDic = copy.deepcopy(self.dic)

        newDic.update({"dx_predict": self.dx_predict,
                       "dy_predict": self.dy_predict,
                       "xieFoV": self.xie, 
                       "xibFoV": self.xib,
                       "xieGP": self.xieGP, 
                       "xibGP": self.xibGP,
                       "xieResiduals": self.xieResiduals, 
                       "xibResiudals": self.xibResiduals,
                       "logr": self.logr,})
        newPkl = open(os.path.join(self.repOut, f"{self.dic["visit"]}_gpOutput.pkl"), 'wb')
        pickle.dump(newDic, newPkl)
        newPkl.close()


                    

if __name__ == "__main__":

    gp = goGPgo(pklIn='../data/visits_w23/2025052000330.pkl',
                repOut='../data/visits_w23_GP_OUT',
                repPlot='../plots',)
    gp.run_gp()
    gp.plotThemAll()
    gp.write_output()