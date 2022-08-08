function n = specify_lens_index(phase, n)
    if(nargin < 2)
        n = 16;
    end
    thre = 2 * pi / 16;
    % thre = 0;
    anchor = linspace(0, -2*pi, 17) + thre / 2;
    
    if(phase <= anchor(n))
        return
    else
        if(n == 0 )
            return
        end
        n = specify_lens_index(phase, n-1);
    end

end