
clc, clear, close all

%% 1. make analytic phantom, 
load('figure2_simulation_roi.mat'),
maskall = sum(squeeze(roi(:,:,30,:)),3);
maskall2 = maskall(:,5:84);

xres = 80;
yres = 88;
zres = 10;

iWF1 = [ 0.02:0.02:0.24,       0.10*ones(1,12)     ];
iWF2 = [ 1-[0.02:0.02:0.24],   1-0.10*ones(1,12)   ];

iT21 = [10*ones(1,18) 5:2:15];
iT22 = [60*ones(1,12) 40:10:90 60*ones(1,6)];

WF1 = zeros([xres,yres,zres]);
WF2 = zeros([xres,yres,zres]);

T21 = zeros([xres,yres,zres]);
T22 = zeros([xres,yres,zres]);

roi = roi(:,:,17:17+zres-1,:);
ord_roi = vec(reshape(1:24,[4 6])')';
for rr = 1 : size(roi,4)
    k = ord_roi(rr);
    
    p1 = iWF1(rr);
    p2 = iWF2(rr);
    
    WF1(roi(:,:,:,k)==1) = p1;
    WF2(roi(:,:,:,k)==1) = p2;

    T21(roi(:,:,:,k)==1) = iT21(rr);
    T22(roi(:,:,:,k)==1) = iT22(rr);
end

% plot ground truth
MWFtrue = WF1./(WF1+WF2);
figure, imagesc(MWFtrue(:,5:84,1)), axis off image, colormap hot, caxis([0 0.3]),colorbar, title('True MWF')
figure, imagesc(T21(:,5:84,1)), axis off image, colormap gray, caxis([0 20]),colorbar, title('True T2* fast')
figure, imagesc(T22(:,5:84,1)), axis off image, colormap gray, caxis([20 100]),colorbar, title('True T2* slow')


%% 2. make distributed T2*, logscale,
[xres yres zres] = size(WF1);
te = (0 : 1 : 63)'+ 2;
te = te - 1;
te = permute(te, [2 3 4 1]);
te = repmat(te,[xres,yres,zres,1]);
eres = size(te,4);

MWF_s1 = WF1(:,:,1);
T2S_s1 = T21(:,:,1);
T2L_s1 = T22(:,:,1);
te_s1 = squeeze(te(:,:,1,:));
t2length=120;
t2s = logspace( log10( 1.5 ), log10( 300 ), t2length );

T2S_dist = zeros(xres,yres,t2length);
T2L_dist = zeros(xres,yres,t2length);
T2_dist = zeros(xres,yres,t2length);
for xx = 1:xres
    for yy = 1:yres
        if T2S_s1(xx,yy) > 0
        temp1 = normpdf(t2s,T2L_s1(xx,yy),T2L_s1(xx,yy)*0.1); 
        temp2 = normpdf(t2s,T2S_s1(xx,yy),T2S_s1(xx,yy)*0.1);
        T2L_dist(xx,yy,:) = temp1/sum(temp1);
        T2S_dist(xx,yy,:) = temp2/sum(temp2);
        else
        T2L_dist(xx,yy,:) = 0;
        T2S_dist(xx,yy,:) = 0;            
        end
        %xres yres t2res
        T2_dist(xx,yy,:) =  MWF_s1(xx,yy)*T2S_dist(xx,yy,:) + (1-MWF_s1(xx,yy))*T2L_dist(xx,yy,:);
    end
end

t2s_rep = repmat(permute(t2s,[1 3 2]),xres,yres,1);
comp_sig = zeros(80,88,eres);
for xx = 1:xres
    for yy = 1:yres
        for tett = 1:eres
            comp_sig(xx,yy,tett) = sum(T2_dist(xx,yy,:) .* exp(-te_s1(xx,yy,tett)./t2s_rep(xx,yy,:))); 
        end
    end
end


%% 3.noise addition
sig_xyzt = repmat(permute(comp_sig,[1,2,4,3]),1,1,zres,1);
sig_xyzt = sig_xyzt./max(sig_xyzt(:));
Nx=xres; Ny=yres; Nz=zres; Nt=64;
snr_in = [300:-20:60].'/1.666666; 
Nsnr = length(snr_in);
noi_matrix = zeros(Nx, Ny, Nz, Nt, Nsnr);
for nn = 1:Nsnr
    noi_matrix(:,:,:,:,nn) = randn(Nx, Ny, Nz, Nt)./snr_in(nn);
end
sig_xyzt_noi = zeros(Nx, Ny, Nz, Nt, Nsnr);
for nn = 1:Nsnr
    sig_xyzt_noi(:,:,:,:,nn) = abs(sig_xyzt(:,:,:,:)+noi_matrix(:,:,:,:,nn));
end
noi_std = zeros(Nsnr,1);
for nn = 1:Nsnr
    noi_std(nn) = std(vec(abs(noi_matrix(:,:,:,1,nn))));
end
snr_out = zeros(1,Nsnr).';
for nn = 1:Nsnr
    snr_out(nn) = mean(vec(sig_xyzt_noi(34:34+3,18:18+3,1,1,nn)))./std(vec(sig_xyzt_noi(3:13,3:13,1,1,nn)));
end
sig_xyzt_noi2 = sig_xyzt_noi(:,:,:,1:32,:);
[xres yres zres eres Nsnr] = size(sig_xyzt_noi2);


%% 4. magnitude model-based MWF
te_m3c = squeeze(te_s1(1,1,1:32))/1000;
rmap_xyzfs = zeros(xres,yres,zres,7,Nsnr);
for nsnr = 6:6
for sn = 1:1
        rmap_xyf = zeros(xres,yres,7);

        im_xye = squeeze(sig_xyzt_noi2(:,:,sn,:,nsnr));
        [xres yres eres] = size(im_xye);
        im_xye(isnan(im_xye)) = 0;
        im_xye(isinf(im_xye)) = 0;        
        
        mask = squeeze(sqrt(sum(abs(sig_xyzt_noi2(:,:,sn,1,nsnr)).^2,4)));
        mask(mask<mean(mask(:))) = 0;
        mask(mask>0) = 1;
        mask0 = reshape(squeeze(mask(:,:)),xres*yres,1);
        
        ten = te_m3c(1:eres);             
        tolvalue=1e-5;
        options = optimset('display','off','Tolx',tolvalue,'TolFun',tolvalue);
        
        sig_tmp = abs(reshape(squeeze(double((im_xye(:,:,1:eres)))),xres*yres,eres));
        rmap_xy = zeros(xres*yres,7);   
        
        parfor yy = 1:yres*xres
            if mask0(yy)>0
                sig = (squeeze(sig_tmp(yy,1:eres)))';
                
                lvf = double([0   0   0     1/.024  1/.150  1/.150]);
                dvf = double([.1 .6  .3     1/.010  1/.064  1/.048]);
                uvf = double([2   2   2     1/.003  1/.024  1/.024]);        
                
                [xc lsq] = lsqnonlin(@(x)tfunc_m3c(x,double(ten),double(sig/abs(sig(1)))),dvf,lvf,uvf,options);
                
                xcomp = xc(1:6);
                xcomp([1 2 3]) = xcomp([1 2 3])*abs(sig(1));        
                sigfc = real(xcomp(1))*exp(-ten*xcomp(4)) + real(xcomp(2))*exp(-ten*xcomp(5)) + real(xcomp(3))*exp(-ten*xcomp(6));
                lsq = sqrt(sum((abs(sigfc)-abs(sig)).^2));
                
                rmap_xy(yy,:)=[xcomp lsq]; 
            end
        end
        rmap_xyf(:,:,:) = reshape(rmap_xy,xres,yres,7);        
        rmap_xyzfs(:,:,sn,:,nsnr) = rmap_xyf;
end
end

zsl = 1;
IWm3c = squeeze(rmap_xyzfs(:,:,zsl,3,:));
AWm3c = squeeze(rmap_xyzfs(:,:,zsl,2,:));
MWm3c = squeeze(rmap_xyzfs(:,:,zsl,1,:));
MWFm3c = MWm3c./(IWm3c+AWm3c+MWm3c);

%remove Nan, Inf
MWFm3c(isnan(MWFm3c)) = 0; MWFm3c(isinf(MWFm3c)) = 0;

%roi anlysis for square boxes
for nn = 1:Nsnr
for rr = 1:24
     temp1 = sort(vec(MWFm3c(:,:,nn).*roi(:,:,zsl,rr)),'descend');
     MWFm3c_roi_mean(rr,nn) = mean(temp1(1:64))*100;
     MWFm3c_roi_std(rr,nn) = std(temp1(1:64))*100;   
end
end

%plot
figure,
for ss = 6:6
subplot(1,2,1*2-1), errorbar(MWFm3c_roi_mean([1:4:21,2:4:22],ss),MWFm3c_roi_std([1:4:21,2:4:22],ss)), hold on, plot([1:1:12],[2:2:24]),
set(gca,'LineWidth',1,'XTick',[0:1:13],'YTick',[0:1:13]*2), grid off, axis square, axis([0 13 0 26]), title('Magnitude model-based MWF, ROI#1~12');
subplot(1,2,1*2), errorbar(MWFm3c_roi_mean([3:4:23,4:4:24],ss),MWFm3c_roi_std([3:4:23,4:4:24],ss)), hold on, plot([1:1:12],ones(1,12)*10);
set(gca,'LineWidth',1,'XTick',[0:1:13],'YTick',[0:1:13]*2), grid off, axis square, axis([0 13 0 26]), title('Magnitude model-based MWF, ROI#13~24');
end
figure, imagesc(MWFm3c(:,5:84,6).*maskall2), axis off image, colormap hot, caxis([0 0.3]),colorbar, title('Magnitude model-based MWF map')


%% 5. rPCA-MWF
sn=1;
sig_xyzt_noi2 = sig_xyzt_noi(:,:,:,1:32,:);
[xres yres zres eres Nsnr] = size(sig_xyzt_noi2);
sig_rpca = sig_xyzt_noi2;
te_rpca = repmat(reshape(te_s1,[80,88,1,64]),1,1,10,1,Nsnr);
temp_rpca2 = zeros(xres,yres,zres,eres,3,Nsnr);

for nn = 6:6    
im_temp = abs(sig_rpca(:,:,:,1:1:32,nn));
te_rpca2 = te_rpca(:,:,:,1:1:32,nn)*1e-3;

delta =  1e-2;       
delta1 = 1e-2;       
num_iter = 1e2;        
tol_update = 1e-4;  
nlr = [1 1 10];     
sth = 0.0;           
scale =  3;         
L_hank = 16;                      
nb = 4;              
nr2 = 2;

N = size(im_temp);
np = prod(N)/N(4);
nplr = prod(nlr);

% make mask for minimization
msk = abs(im_temp(:,:,:,1)); msk(msk<0.1*max(msk(:))) = 0; msk(msk~=0) = 1;
msk = repmat(msk,[1 1 1 N(4)]);

cmp = 1*exp(1000*te_rpca2/scale).^1;
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
            [uu,ss,vv] = svd(qtmp_hankel);
            [W,H] = seminmf(qtmp_hankel, nb);
            sol = W(:,1:1) * H(1:1,:);
            L(xp,yp,zp,:) = reshape(pons_matrix(sol,(N(4)-L_hank+1)),...
                [nlr(1) nlr(2) nlr(3) N(4)]) ./ cmp(xp,yp,zp,:);
        end
    end
end
L2 = im_temp - L;

X(:,:,:,:,1) = L;  
X(:,:,:,:,2) = L2*ss(1,1)./ss(2,2);     
X(:,:,:,:,3) = 0;           

Z = X.*0;
U = X.*0;

L_hankel1 = L_hank;
hsize1 = (N(4)-L_hankel1+1);
L_hankel2 = L_hank;
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
    L1_rpca = temp(:,:,5,1,1,nn);
    L2_rpca = temp(:,:,5,1,2,nn);
    S_rpca = temp(:,:,5,1,3,nn);
end

L1_rpca = squeeze(L1_rpca);
L2_rpca = squeeze(L2_rpca);
Srpca = squeeze(S_rpca);
MWFrpca = L2_rpca./(L1_rpca+L2_rpca);

%remove Nan, Inf
MWFrpca(isnan(MWFrpca)) = 0; MWFrpca(isinf(MWFrpca)) = 0;

%roi anlysis for square boxes
for nn = 1:Nsnr
for rr = 1:24
     temp1 = sort(vec(MWFrpca(:,:,nn).*roi(:,:,zsl,rr)),'descend');
     MWFrpca_roi_mean(rr,nn) = mean(temp1(1:64))*100;
     MWFrpca_roi_std(rr,nn) = std(temp1(1:64))*100;   
end
end

%plot
figure,
for ss = 6:6
subplot(1,2,1*2-1), errorbar(MWFrpca_roi_mean([1:4:21,2:4:22],ss),MWFrpca_roi_std([1:4:21,2:4:22],ss)), hold on, plot([1:1:12],[2:2:24]),
set(gca,'LineWidth',1,'XTick',[0:1:13],'YTick',[0:1:13]*2), grid off, axis square, axis([0 13 0 26]), title('rPCA-MWF, ROI#1~12');
subplot(1,2,1*2), errorbar(MWFrpca_roi_mean([3:4:23,4:4:24],ss),MWFrpca_roi_std([3:4:23,4:4:24],ss)), hold on, plot([1:1:12],ones(1,12)*10);
set(gca,'LineWidth',1,'XTick',[0:1:13],'YTick',[0:1:13]*2), grid off, axis square, axis([0 13 0 26]), title('rPCA-MWF, ROI#13~24');
end
figure, imagesc(MWFrpca(:,5:84,6).*maskall2), axis off image, colormap hot, caxis([0 0.3]),colorbar, title('rPCA-MWF map')


