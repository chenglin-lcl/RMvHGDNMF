function [V_star, obj, error_cnt] = RMvHGDNMF_update(X, layers, varargin)
% Input: X(M*nSmp)
% layers >= 2
warning('off');
pnames = {'maxiter', 'tolfun', 'lambda', 'gamma', 'normalize'};
dflts  = {500, 1e-5, 0, 0, 1};
[maxiter, tolfun, lambda, gamma, normalize] = internal.stats.parseArgs(pnames,dflts,varargin{:});

view_num = length(X); % 视图个数
layer_num = length(layers); % 分解层数

U = cell(view_num, layer_num);
V = cell(view_num, layer_num);
V_err = cell(view_num, layer_num);
obj = zeros(1, 1);

% 数据进行归一化
if normalize == 1
    for i = 1: view_num
        [X{i}, ~] = data_normalization(X{i}, [], 'std');
    end
end

% 构建超图
L = cell(1, view_num); % 超拉普拉斯
W = cell(1, view_num); % 相似度矩阵
W1 = cell(1, view_num); % 度矩阵
param.k = 5;
for view_idx = 1: view_num
    HG = gsp_nn_hypergraph(X{view_idx}', param); % 输入是n*m
    % L = Dv - S
    L{view_idx} = HG.L;
    W1{view_idx} = diag(HG.dv);
    % S = Dv - L
    W{view_idx} = W1{view_idx} - L{view_idx};
end

% 预训练
for view_idx = 1: view_num
    for layer_idx = 1: layer_num
        if layer_idx == 1
           upper_V = X{view_idx}; 
        else
           upper_V = V{view_idx, layer_idx-1};
        end
        % 使用kmeans进行初始化
        [U{view_idx, layer_idx}, V{view_idx, layer_idx}] = KMeansdata(upper_V, layers(layer_idx)); 
    end    
end


% 更新错误计数器
error_cnt = 0;

% 迭代更新
for iter = 1: maxiter
   for view_idx = 1: view_num

       % V_err: reconstruction of the ith layer's coefficient matrix.
       V_err{view_idx, layer_num} = V{view_idx, layer_num};
       for layer_idx = layer_num-1: -1: 1
           V_err{view_idx,layer_idx} = U{view_idx, layer_idx+1} * V_err{view_idx, layer_idx+1};
       end
       
       for layer_idx = 1: layer_num
           if layer_idx == 1
               % update D
               E = X{view_idx}-U{view_idx, 1} * V_err{view_idx,1};
               D = sparse(get_weight_matrix(E));
               % update U_l^p (l=1)
               tempUp = X{view_idx}*D*(V_err{view_idx,1}');
               tempDown = U{view_idx, 1}*V_err{view_idx,1}*D*(V_err{view_idx,1}');
               U{view_idx, 1} = U{view_idx, 1} .* (tempUp ./ max(tempDown, 1e-9));
           else
               % update D
               E = X{view_idx}-PHI * U{view_idx, layer_idx} * V_err{view_idx,layer_idx};
               D = sparse(get_weight_matrix(E));
               % update U^p_l (l>1)
               tempUp = (PHI')*X{view_idx}*D*(V_err{view_idx,layer_idx}');
               tempDown = (PHI')*PHI*U{view_idx, layer_idx}*V_err{view_idx,layer_idx}*D*(V_err{view_idx,layer_idx}');
               U{view_idx, layer_idx} = U{view_idx, layer_idx} .* (tempUp ./ max(tempDown, 1e-9));
           end
           
            % update PHI
            if layer_idx == 1
                PHI = U{view_idx, 1};
            else
                PHI = PHI * U{view_idx, layer_idx};
            end
            
            % update V^p_l
            % update D
            if layer_idx == layer_num
                %update V^p_l (l=L)
                E = X{view_idx}-PHI * V{view_idx,layer_idx};
                D = sparse(get_weight_matrix(E));
                tempUp = (PHI')*X{view_idx}*D + lambda*sum_Vq(V, view_num, view_idx, layer_num) + gamma*V{view_idx, layer_idx}*W{view_idx};
                tempDown = (PHI')*PHI*V{view_idx,layer_idx}*D + lambda*(view_num-1)*V{view_idx, layer_idx} + gamma*V{view_idx, layer_idx}*W1{view_idx};
                V{view_idx,layer_idx} = V{view_idx,layer_idx} .* (tempUp ./ max(tempDown, 1e-9)); 
            end

       end
       
   end

   obj(iter) = Calc_obj_value(X, U, V, L, lambda, gamma, layers);
   fprintf('iter = %d, obj = %g\n', iter, obj(iter));
   if (iter>=2)&&(obj(iter)>obj(iter-1))
      error_cnt = error_cnt + 1; 
   end
    if (iter>2) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<tolfun)|| iter==maxiter
        break;
    end
   
end
V_star = zeros(size(V{view_num, layer_num}));
for view_idx = 1: view_num
    V_star = V_star + V{view_idx, layer_num};
end
V_star = V_star/view_num;
[V_star, ~] = data_normalization(V_star, [], 'std');

end

function [obj_value] = Calc_obj_value(X, U, V, L, lambda, gamma, layers)
view_num = length(X); % 视图个数
layer_num = length(layers); % 分解层数
% log损失
obj_value = l21_NMF_cost(X, U, V, view_num, layer_num);
for view_idx = 1: view_num
    % 顶层图正则化
    obj_value = obj_value...
        + lambda*sum_Vp_Vq(V, view_num, view_idx, layer_num)...
        + gamma*trace(V{view_idx, layer_num}*L{view_idx}*(V{view_idx, layer_num}'));
end
end

function [obj_value] = l21_NMF_cost(X, U, V, view_num, layer_num)
obj_value = 0;
for view_idx = 1: view_num
    E = X{view_idx} - reconstruction(U, V, layer_num, view_idx);
    obj_value = obj_value + sum(sqrt(sum((E).^2, 1)));
end
end

function [out] = reconstruction(U, V, layer_num, view_idx)
% out 当前视图的重建矩阵
out = V{view_idx, layer_num};
for k = layer_num : -1 : 1
    out =  U{view_idx, k} * out;
end
end

function [D] = get_weight_matrix(E)
Ei = max(sqrt(sum(E.^2, 1)), 1e-9);
D = diag(1./(2*Ei));
end

function [res] = sum_Vq(V, view_num, p, layer_num)
    res = zeros(size(V{p, layer_num}));
    for q = 1: view_num
       if q ~= p
          res = res + V{q, layer_num};
       end
    end
end

function [obj] = sum_Vp_Vq(V, view_num, p, layer_num)
    obj = 0;
    for q = 1: view_num
        if q ~= p
            obj = obj + (norm(V{p, layer_num} - V{q, layer_num}, 'fro').^2);
        end
    end
end



