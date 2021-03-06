<<<<<<< HEAD
%ComputeMarginal Computes the marginal over a set of given variables
%   M = ComputeMarginal(V, F, E) computes the marginal over variables V
%   in the distribution induced by the set of factors F, given evidence E
%
%   M is a factor containing the marginal over variables V
%   V is a vector containing the variables in the marginal e.g. [1 2 3] for
%     X_1, X_2 and X_3.
%   F is a vector of factors (struct array) containing the factors 
%     defining the distribution
%   E is an N-by-2 matrix, each row being a variable/value pair. 
%     Variables are in the first column and values are in the second column.
%     If there is no evidence, pass in the empty matrix [] for E.


function M = ComputeMarginal(V, F, E)

% Check for empty factor list
if (numel(F) == 0)
      warning('Warning: empty factor list');
      M = struct('var', [], 'card', [], 'val', []);      
      return;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE:
% M should be a factor
% Remember to renormalize the entries of M!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   M = struct('var', [], 'card', [], 'val', []); % Returns empty factor. Change this.
    
    J = ComputeJointDistribution(F);
    O = ObserveEvidence(J, E);
    
    [M.var Oi Vi] = intersect(O.var, V);
    [dummy, mapM] = ismember(M.var, O.var);
    M.card = O.card(Oi);
    M.val = zeros(1,prod(M.card));

    for k = 1:length(O.val),
        AO = IndexToAssignment(k, O.card);
        AM = AO(mapM);
        idx = AssignmentToIndex(AM, M.card);
        M.val(idx) = M.val(idx) + O.val(k);
    end;    
    total = sum(M.val);
	M.val = M.val/total;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
=======
%ComputeMarginal Computes the marginal over a set of given variables
%   M = ComputeMarginal(V, F, E) computes the marginal over variables V
%   in the distribution induced by the set of factors F, given evidence E
%
%   M is a factor containing the marginal over variables V
%   V is a vector containing the variables in the marginal e.g. [1 2 3] for
%     X_1, X_2 and X_3.
%   F is a vector of factors (struct array) containing the factors 
%     defining the distribution
%   E is an N-by-2 matrix, each row being a variable/value pair. 
%     Variables are in the first column and values are in the second column.
%     If there is no evidence, pass in the empty matrix [] for E.


function M = ComputeMarginal(V, F, E)

% Check for empty factor list
if (numel(F) == 0)
      warning('Warning: empty factor list');
      M = struct('var', [], 'card', [], 'val', []);      
      return;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE:
% M should be a factor
% Remember to renormalize the entries of M!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   M = struct('var', [], 'card', [], 'val', []); % Returns empty factor. Change this.
    
    J = ComputeJointDistribution(F);
    O = ObserveEvidence(J, E);
    
    [M.var Oi Vi] = intersect(O.var, V);
    [dummy, mapM] = ismember(M.var, O.var);
    M.card = O.card(Oi);
    M.val = zeros(1,prod(M.card));

    for k = 1:length(O.val),
        AO = IndexToAssignment(k, O.card);
        AM = AO(mapM);
        idx = AssignmentToIndex(AM, M.card);
        M.val(idx) = M.val(idx) + O.val(k);
    end;    
    total = sum(M.val);
	M.val = M.val/total;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
>>>>>>> f00d97771ad9dc8e6c9e9ff582ecedac656678ca
