function C = colCellMatCat(A, varargin)
   % A must be an N x 1 cell array
   % elements of A must be matrices with matching dim
   if nargin == 2
       dim = varargin{:};
   else
       dim = 1;
   end

   C = cell(size(A{1}));
   Atmp = vertcat(A{:});

   for k = 1:numel(C)
       C{k} = cat(dim, Atmp{:, k});
   end
end