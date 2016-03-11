function[value,counter] = collatz(limit)

    counter=1;
    value = 1;
    i = limit;
    while i > 1
        j = i;
        count = 0;
        while j > 1
            if mod(j,2)==0
                j = j/2;
                count = count+1;
            else
                j = j*3+1;
                count = count+1;
            end
        end
        if count > counter;
            counter = count;
            value = i;
        end
        i = i-1;
    end
counter;
value;
end