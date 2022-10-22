function [Bs, Bt, As, At, Wst, term_all] = ATH_U(Xs,  Xt_train,  param)
    r = param.r;
    alpha1 = param.alpha1;      alpha2 = param.alpha2; 
    beta1 = param.beta1;        beta2 = param.beta2;
    lambda2 = param.lambda2;
    regu1 = param.regu1;        regu2 = param.regu2;
    T = param.T;
    [ns, ds] = size(Xs);
    [nt, dt] = size(Xt_train);
    I_ns = eye(ns, ns);
    I_ds = eye(ds, ds);
    I_dt = eye(dt, dt);
    I_nt = eye(nt, nt);
    neighK = 10;
    options = [];
    options.NeighborMode = 'KNN';
    options.WeightMode = 'Binary';
    options.k = neighK;      
    Ws = constructW(Xs,options);    
    Ds = diag(sum(Ws));
    Ls = Ds - Ws;
     
    neighK = 10;
    options = [];
    options.NeighborMode = 'KNN';
    options.WeightMode = 'Binary';
    options.k = neighK;      
    Wt = constructW(Xt_train,options);    
    Dt = diag(sum(Wt));
    Lt = Dt - Wt;
    

    
    %% initialization
    [As, ~] = PCA_dencai(Xs, r); 
    [At, ~] = PCA_dencai(Xt_train, r); 
    ini_Bs = Xs * As;
    ini_Bt = Xt_train * At;
    Bs = sign(ini_Bs');
    Bt = sign(ini_Bt');
    
    

    

    Xs = Xs';
    Xt_train = Xt_train';
    term_all = zeros(T, 1);
    

    k = neighK;
    distX = L2_distance_1(As' * Xs, At' * Xt_train);
    [distX1, idx] = sort(distX, 2);
    gama0 = zeros(ns, 1);
    for i = 1:ns
        di = distX1(i, 2:k+2);
        gama0(i) = 0.5 * (k * di(k+1) - sum(di(1:k)));
    end
    gama = mean(gama0);


    for ii = 1:T                             
         
       %% update Bs  Bt
        Ms = alpha1 * As' * Xs ;  
        Mt = alpha2 * At' * Xt_train; 
        for i = 1:r
            [~,indS]=sort(Ms(i,:));
            firsthalfS=indS(1:round(ns/2));
            secondhalfS=indS(round(ns/2)+1:end);
            Bs(i, firsthalfS)  = -1; 
            Bs(i, secondhalfS) = 1;
            
            [~,indT]=sort(Mt(i,:));
            firsthalfT=indT(1:round(nt/2));
            secondhalfT=indT(round(nt/2)+1:end);
            Bt(i, firsthalfT)  = -1;
            Bt(i, secondhalfT) = 1;
        end 
        
        %% update Wst
        distx = L2_distance_1(As' * Xs, At' * Xt_train); 
        if ii>5%1
            [~, idx] = sort(distx, 2);
        end
        Wst = zeros(ns, nt);
        for i=1:ns
            idxa0 = idx(i,2:k+1);
            dxi = distx(i,idxa0);   
            ad = -(dxi) / (2 * gama);
            Wst(i,idxa0) = EProjSimplex_new(ad);
        end  
        Dsts = diag(sum(Wst,2)); 
        Dstt = diag(sum(Wst,1));          
        
       %% update As  At
        As = (Xs * (alpha1 * I_ns + beta1 * Ls + lambda2 * Dsts) * Xs' + regu1 * I_ds) \ (alpha1 * Xs * Bs' + lambda2 * Xs * Wst * Xt_train' * At);  
        At = (Xt_train * (alpha2 * I_nt + beta2 * Lt + lambda2 * Dstt) * Xt_train' + regu2 * I_dt) \ (alpha2 * Xt_train * Bt' + lambda2 * Xt_train * Wst' * Xs' * As);  

        
        term1 = alpha1 * norm((Bs-As'*Xs),'fro')^2 + alpha2 * norm((Bt-At'*Xt_train),'fro')^2;
        term2 = beta1 * trace(As'*Xs*Ls*Xs'*As) + beta2 * trace(At'*Xt_train*Lt*Xt_train'*At);
        term3 = lambda2 * trace(As'*Xs*Dsts*Xs'*As + At'*Xt_train*Dstt*Xt_train'*At - 2*As'*Xs*Wst*Xt_train'*At);
        term4 = gama * norm((Wst),'fro')^2 ;
        term = term1 + term2 + term3 + term4;    
        term_all(1+ii) = term;  

        
    end
end
