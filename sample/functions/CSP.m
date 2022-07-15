% ----------------------------------------------------------------------- %
%                           H    Y    D    R    A                         %
% ----------------------------------------------------------------------- %
% Function 'csp' trains a Common Spatial Pattern (CSP) filter bank.       %
%                                                                         %
%   Input parameters:                                                     %
%       - X1:   Signal for the positive class, dimensions [C x T], where  %
%               C is the no. channels and T the no. samples.              %
%       - X2:   Signal for the negative class, dimensions [C x T], where  %
%               C is the no. channels and T the no. samples.              %
%                                                                         %
%   Output variables:                                                     %
%       - W:        Filter matrix (mixing matrix, forward model). Note that
%                   the columns of W are the spatial filters.             %
%       - lambda:   Eigenvalues of each filter.                           %
%       - A:        Demixing matrix (backward model).                     %
% ----------------------------------------------------------------------- %
%   Versions:                                                             %
%       - 1.0:     (19/07/2019) Original script.                          %
% ----------------------------------------------------------------------- %
%   Script information:                                                   %
%       - Version:      1.0.                                              %
%       - Author:       V. Martínez-Cagigal                               %
%       - Date:         19/07/2019                                        %
% ----------------------------------------------------------------------- %
%   Example of use:                                                       %
%       csp_example;                                                      %
% ----------------------------------------------------------------------- %
%   References:                                                           %
%       [1]     Blankertz, B., Tomioka, R., Lemm, S., Kawanabe, M., &     %
%               Muller, K. R. (2007). Optimizing spatial filters for robust 
%               EEG single-trial analysis. IEEE Signal processing magazine, 
%               25(1), 41-56.                                             %
% ----------------------------------------------------------------------- %
function [W, lambda, A] = CSP(X1, X2)
    % Error detection
    if nargin < 2, error('Not enough parameters.'); end
    if ndims(X1) ~= ndims(X2)
        error('The dimension of trial signals must be same');
    end
    
    if ndims(X1) > 4
        error('The size of trial signals must be [C x T] or [Epoch x C x T]');
    end
    
    % Compute the covariance matrix of each class
    if ismatrix(X1)
        S1 = cov(X1');   % S1~[C x C]
        S2 = cov(X2');   % S2~[C x C]
    else
        S1 = zeros(size(X1, 1), size(X1, 2), size(X1, 2));
        S2 = zeros(size(S1));
        
        for i = 1:size(X1, 1)
            S1(i, :, :) = cov(squeeze(X1(i, :, :))');
            S2(i, :, :) = cov(squeeze(X2(i, :, :))');
        end
        S1 = squeeze(mean(S1, 1));
        S2 = squeeze(mean(S2, 1));
    end
    
    % Solve the eigenvalue problem S1·W = l·S2·W
    [W,L] = eig(S1, S1 + S2);   % Mixing matrix W (spatial filters are columns)
    lambda = diag(L);           % Eigenvalues
    A = (inv(W))';              % Demixing matrix
    
    % Further notes:
    %   - CSP filtered signal is computed as: X_csp = W'*X;
end