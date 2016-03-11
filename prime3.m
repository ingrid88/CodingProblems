function[prime] = prime3(num)

i=2;
number = num;
while num > 1
   if mod(num,i)==0
       num = num/i;
       prime = i;
   else
       i=i+1;
   end
end
end