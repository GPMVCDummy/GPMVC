function [ac] = accuracy(T,H)
      if length(T) ~= length(H)
        size(T)
        size(H)
      end    
      
      num = length(T);
      ac = 0;
      for i=1:length(T)
      	if (T(i)==H(i))
      		ac = ac + 1;
      	end
      end
      ac = ac/num;     	
