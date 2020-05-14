import matplotlib
import numpy as np
from scipy.stats import *
import scipy as sc
import math

def main():
    f = '/home/serene/PycharmProjects/multimodaltraj/kernel_models/MX-LSTM-master/data/annotation_tc.txt'
             # 'r')

    mat = np.genfromtxt(f, delimiter=',')
    headPose = mat[:,12]
    loc_x = np.power(np.subtract(mat[:,8], mat[:,10]), 2.0)
    loc_y = np.power(np.subtract(mat[:,9], mat[:,11]), 2.0)

    # MX-LSTM assumes gaussian distribution of trajectories and that is why they use pearson correlation test
    corr, _ = pearsonr(headPose, loc_y)
    print('Linear correlation = ' , corr)

    # For non-Gaussian assumptions
    corr, _ = spearmanr(headPose, loc_y)
    print('Non-Linear correlation = ', corr)
    return

    # Negatively correlated features in both linear and non-linear tests
    #
if __name__ == '__main__':
    main()

# genVisualization = 0;
# counter = 1;
#
# FADErr = [];
# for ii=1:size(data, 1)
# [gtPts, thisDt, frameId, pedId, thisPed, frameInfo, linInd] = extractNomralizedTraj(data, ii);
# idx = find(arrayT(:, 2) == thisPed);
# allPedAnno = arrayT(idx,:);
# dl = diff(allPedAnno(:, 3: 4));
# if (size(dl, 1) > 1)
#     dl = [allPedAnno(1, [3 4]);
#     dl];
#     else
#     dl = [0 0];
# end
# allPedAnno(:, [3 4])=dl;
# if length(linInd) == 20
#     [gp, ph] = denomPts(gtPts, thisDt, normParams, allPedAnno, origFrames, normFrame, frameId);
#     err = mean(sqrt(sum((gp(9:end,:) - ph(9:end,:)).^ 2, 2)));
#     finalErr = mean(sqrt(sum((gp(end,:) - ph(end,:)).^ 2, 2)));
#     MADErr(counter, 1) = err;
#     FADErr(counter, 1) = finalErr;
#     finalErr = [];
#     counter = counter + 1;
#     if genVisualization
#         gtIm = w2i(gp, D.H); % in image
#         plane
#         predIm = w2i(ph, D.H);
#         im = imread(sprintf('%06d.png', frameId(1)));
#         figure
#         imshow(im), hold
#         on
#         plot(gtIm(:, 1), gtIm(:, 2), 'g*')
#         plot(predIm(9: end, 1), predIm(9: end, 2), 'b*')
#         pause(0.8)
#     end
# end
#
# end