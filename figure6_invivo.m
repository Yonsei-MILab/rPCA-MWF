clc,clear,close all 

% load mGRE data
load('figure6_invivo_mGRE.mat'),


% plot complex model-based MWF
load('figure6_complex_MWF.mat'),
MWFc = complex_model_based_MWF;
MWFc(isnan(complex_model_based_MWF))=0; MWFc(isinf(complex_model_based_MWF))=0;
figure, imshow3(MWFc,[0 0.3],[4 10],'hot'); colorbar; axis off image;title('MWFc');


% rPCA-MWF
mGRE = permute(abs(mGRE),[1 2 3 4]);
figure, imshow3(mGRE(:,:,:,1),[0 1],[4 10],'gray'); axis off image; title('1st echo mGRE');
figure, imshow3(mGRE(:,:,:,end),[0 1],[4 10],'gray'); axis off image; title('30th echo mGRE');
te_3d = repmat(reshape(te,1,1,1,30),128,128,30,1);

im_temp = double(abs(mGRE(:,:,1:30,1:2:end,1)));
te_temp = double(te_3d(:,:,:,1:2:end));
N = size(im_temp);          
delta  = 1e-2;         
delta1 = 1e-2;          
num_iter = 100;       
tol_update = 1e-4;   
nlr = [128 128 30]; 
sth = 0.01;         
scale = 2;           
L_hank = 8;                          
nb = 4;               
nr2 = 2;
nplr = prod(nlr);

% make mask for minimization
msk = abs(im_temp(:,:,:,1)); msk(msk<0.1*max(msk(:))) = 0; msk(msk~=0) = 1;
msk = repmat(msk,[1 1 1 N(4)]);

cmp = exp(1000*te_temp/scale);
L = zeros(N);
for iz = 1 : floor(N(3)/nlr(3))
    pos(3) = nlr(3) * (iz-1) + 1;
    zp = pos(3) : (pos(3) + nlr(3) - 1);
    for ix = 1 : floor(N(1)/nlr(1))
        pos(1) = nlr(1) * (ix-1) + 1;
        xp = pos(1) : (pos(1) + nlr(1) - 1);
        for iy = 1 : floor(N(2)/nlr(2))
            pos(2) = nlr(2) * (iy-1) + 1;
            yp = pos(2) : (pos(2) + nlr(2) - 1);
            greC = im_temp(xp,yp,zp,:).*cmp(xp,yp,zp,:);
            qtmp_hankel = genHankel_matrix(reshape(greC ,[nplr,N(4)]), L_hank);
            [W,H] = seminmf(qtmp_hankel, nb);
            sol = W(:,1:1) * H(1:1,:);
            L(xp,yp,zp,:) = reshape(pons_matrix(sol,(N(4)-L_hank+1)),...
                [nlr(1) nlr(2) nlr(3) N(4)]) ./ cmp(xp,yp,zp,:);
        end
    end
end
L2 = im_temp - L;

X(:,:,:,:,1) = L*1;   
X(:,:,:,:,2) = L2*1;          
X(:,:,:,:,3) = 0;           

Z = X.*0;
U = X.*0;

L_hankel1 = L_hank/1;
hsize1 = (N(4)-L_hankel1+1);
L_hankel2 = L_hank/1;
hsize2 = (N(4)-L_hankel2+1);
delta =  5e-1 *50;       
delta1 = 5e-1 *50;    

for t = 1:5        
    if t == 1
        ru = [1 1 1]-1;
    else
        ru = [randi(nlr(1)) randi(nlr(2)) 1]-1;
    end    
    Z = Z.* 0;
    for iz = 1 : floor((N(3) - ru(3))/nlr(3))        
        pos(3) = ru(3) + nlr(3) * (iz-1) + 1;
        zp = pos(3) : (pos(3) + nlr(3) - 1);        
        for ix = 1 : floor((N(1) - ru(1))/nlr(1))            
            pos(1) = ru(1) + nlr(1) * (ix-1) + 1;
            xp = pos(1) : (pos(1) + nlr(1) - 1);            
            for iy = 1 : floor((N(2) - ru(2))/nlr(2))                
                pos(2) = ru(2) + nlr(2) * (iy-1) + 1;
                yp = pos(2) : (pos(2) + nlr(2) - 1);                
                qtmp_hankel = genHankel_matrix(reshape((X(xp,yp,zp,:,1) + 1/delta * U(xp,yp,zp,:,1))...
                    ,[nplr,N(4)]) , L_hankel1);
                [u,s,v] = svd(qtmp_hankel,0);
                Z(xp,yp,zp,:,1) = reshape(pons_matrix(u(:,1:1) * s(1:1,1:1) * v(:,1:1)',hsize1)...
                    ,[nlr(1) nlr(2) nlr(3) N(4)]);
                qtmp_hankel2 = genHankel_matrix(reshape((X(xp,yp,zp,:,2) + 1/delta1 * U(xp,yp,zp,:,2))...
                    ,[nplr,N(4)]) , L_hankel2);
                [u,s,v] = svd(qtmp_hankel2,0);
                Z(xp,yp,zp,:,2) = reshape(pons_matrix(u(:,1:2) * s(1:2,1:2) * v(:,1:2)',hsize2)...
                    ,[nlr(1) nlr(2) nlr(3) N(4)]);
            end
        end
    end    
    Z(:,:,:,:,3) = shrinkr(X(:,:,:,:,3) + 1/delta * U(:,:,:,:,3),sth);    
    if t ~=0
        X_prev = X;
        X(:,:,:,:,1) = ((im_temp - X_prev(:,:,:,:,2) - X_prev(:,:,:,:,3))...
            - U(:,:,:,:,1) + delta * Z(:,:,:,:,1))/ (1 + delta);
        X(:,:,:,:,2) = ((im_temp - X(:,:,:,:,1) - X_prev(:,:,:,:,3))...
            - U(:,:,:,:,2) + delta1 * Z(:,:,:,:,2))/ (1 + delta1);
        X(:,:,:,:,3) = ((im_temp - X(:,:,:,:,1) - X(:,:,:,:,2))...
            - U(:,:,:,:,3) + delta * Z(:,:,:,:,3))/ (1 + delta);
        nlr = randi([2 8],1,2);
        nlr = [nlr, 2];
        nplr = prod(nlr);
    end    
    U = U + delta * (X - Z);           
    x_update = 100 * norm(X(:)-X_prev(:)) / norm(X_prev(:));
    if x_update < tol_update
        break
    end
end
temp = real(X); temp(isnan(temp)) = 0; temp(isinf(temp)) = 0; temp(temp<0) = 0;
L1_rpca = temp(:,:,:,1,1);
L2_rpca = temp(:,:,:,1,2);
S_rpca = temp(:,:,:,1,3);

MWFrpca = L2_rpca./(L1_rpca+L2_rpca);
msk = abs(im_temp(:,:,:,1)); msk(msk<0.1*max(msk(:))) = 0; msk(msk~=0) = 1;

figure; imshow3(MWFrpca(:,:,:,1).* double(msk),[0 0.30],[3 10],'hot'); colorbar; axis off image; title('MWFr');

