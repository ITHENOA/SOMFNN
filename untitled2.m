f=randi(9,10,4)
% for i = 1:5
%     newf(:,:,i) = f((i-1)*2+1:i*2,:);
% end

newf = reshape(f', 4, 2, 5)
newf = pagetranspose(newf)
% Permute dimensions to match the desired order
newf = permute(newf, [2, 1, 3])


 sig=reshape(permute(a,[1 3 2]),wl,1,n,[]);
 bb=reshape(b,wl,1,1,[])
 lam = reshape(l,1,1,n,[])

 k=10
 bs=4
 batch_idx = 1:k;
 for it = 1 : ceil(k/bs)
     batch_idx( (it-1)*bs+1 : min(it*bs,k) )
 end