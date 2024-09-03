function [K,R,c] = decomposeP(P)

% The input P is assumed to be a 3-by-4 homogeneous camera matrix.
% The function returns a homogeneous 3-by-3 calibration matrix K,
% a 3-by-3 rotation matrix R and a 3-by-1 vector c such that
%   K*R*[eye(3), -c] = P.

W = [0 0 1; 0 1 0; 1 0 0];

% calculate K and R (up to sign)
[Qt,Rt] = qr((W*P(:,1:3))');
K = W*Rt'*W;
R = W*Qt';

% correct for negative focal length(s) if necessary
D = [1 0 0; 0 1 0; 0 0 1];
if K(1,1) < 0, D(1,1) = -1; end
if K(2,2) < 0, D(2,2) = -1; end
if K(3,3) < 0, D(3,3) = -1; end
K = K*D;
R = D*R;

% calculate c
c = -R'*inv(K)*P(:,4);

end
