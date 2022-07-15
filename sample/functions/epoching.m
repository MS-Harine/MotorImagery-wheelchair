function epoch = epoching(data, srate, event, range)

    epoch = zeros(length(event), size(data, 1), floor(srate * (range(2) - range(1))));
    
    for i = 1:length(event)
        epoch(i, :, :) = data(:, event(i) + floor(srate * range(1)) + 1 : ...
                                 event(i) + floor(srate * range(2)));
    end

end

