clear;clc;
warning off 
addpath('./data/');
bits = [ 64  ];  %    16   32  48  64  96 128 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    OfficeHome     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
db_name   =   'OfficeHome_CtoA'; 
load ./data/OfficeHome/OfficeHome_CtoA

ns = size(Xs, 1);
nt_train = size(Xt_train, 1);


XX=[Xs;Xt_train];
samplemean = mean(XX,1);
Xs = Xs-repmat(samplemean,size(Xs,1),1);
Xt_train = Xt_train-repmat(samplemean,size(Xt_train,1),1);
Xt_test = Xt_test-repmat(samplemean,size(Xt_test,1),1);

X_train_total = [Xs; Xt_train]; 
Xst = [Xs; Xt_train];
Yst = [Ys; Yt_train];
all_methods = [ "ATH_U"];
for met = 1:length(all_methods)                                                                                                                            
    current_method = all_methods(met) 
    for ii = 1:length(bits)
        bits(ii)
        mAP_cross = zeros(10,1);
        mAP_single = zeros(10,1);
        traintime = zeros(10,1);
        for times = 1:10
            switch(current_method) 
            case 'ATH_U' 
                tic
                method = 'ATH_U';
                param.r = bits(ii);   
                

                param.alpha1 = 10^-2;        param.alpha2 = 10^-1; 
                param.beta1 = 10^-3;        param.beta2 = 10^-1;
                param.lambda2 = 10^0;
                param.regu1 = 10^0;        param.regu2 = 10^-2;
                param.T = 10;
                
                [Bs_inF, Bt_inF, As, At, Wst, term_all] = ATH_U(Xs,  Xt_train,  param);
                
                Bs = (Xs * At > 0);
                Bt_train = (Xt_train * At > 0);
                Bt_test  = (Xt_test * At > 0); 
                traintime(times) = toc;
            otherwise 
                fprintf('No this method!' ); 
            end

           %% cross-domain retrieval
            Bs = compactbit(Bs);
            Bt_test = compactbit(Bt_test);	       
            WtrueTestTraining = compute_S(Ys, Yt_test) ;
            Dhamm = hammingDist(Bt_test, Bs);
            [recall, precision, ~] = recall_precision(WtrueTestTraining', Dhamm);
            mAP_cross(times) = area_RP(recall, precision);
            clear  mAP   Dhamm

           %% single-domain retrieval
            Bt_train = compactbit(Bt_train);      
            WtrueTestTraining = compute_S(Yt_train, Yt_test) ;
            Dhamm = hammingDist(Bt_test, Bt_train);
            [recall, precision, ~] = recall_precision(WtrueTestTraining', Dhamm);
            mAP_single(times) = area_RP(recall, precision);
            clear  mAP  Dhamm
        end
        mAP_cross = sum(mAP_cross) / times;
        mAP_single = sum(mAP_single) / times;
        traintime = sum(traintime) / times;
        if strcmp(method,'ATH_U') 
            save(['results/', method, '_', db_name, '_', num2str(bits(ii)), '_bits.mat'],'Bs', 'Bt_train', 'Bt_test', 'mAP_cross', 'mAP_single', 'traintime', 'param');
        else
            save(['results/', method, '_', db_name, '_', num2str(bits(ii)), '_bits.mat'],'Bs', 'Bt_train', 'Bt_test', 'mAP_cross', 'mAP_single', 'traintime');
        end
    end
end

