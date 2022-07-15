function test_y = classification_csp_lda(train_X, train_y, test_X, varargin)
    
    p = inputParser;
    addRequired(p, 'train_X', @(x) ndims(x) == 3);
    addRequired(p, 'train_y', @isvector);
    addRequired(p, 'test_X', @(x) ndims(x) == 3);
    addOptional(p, 'cspLines', 0, @isscalar);
    parse(p, train_X, train_y, test_X, varargin{:});
    cspLines = p.Results.cspLines;
    
    % Validation
    classes = unique(train_y);
    if length(classes) ~= 2
        error("CSP only support binary classification only");
    end
    
    % CSP
    train_C1 = train_X(train_y == classes(1), :, :);
    train_C2 = train_X(train_y == classes(2), :, :);
    
    csp_filter = CSP(train_C1, train_C2);
    channels = size(csp_filter, 1);
    if cspLines ~= 0
        csp_filter = csp_filter(:, [1:cspLines, channels - cspLines + 1 : channels]);
    end
    
    csp_train = zeros(size(train_X, 1), size(csp_filter', 1));
    for epoch = 1:size(train_X, 1)
        for filter = 1:size(csp_filter', 1)
            csp_train(epoch, filter) = log(csp_filter(:, filter)' * squeeze(train_X(epoch, :, :)) * ...
                                           squeeze(train_X(epoch, :, :))' * csp_filter(:, filter));
        end
    end
    
    csp_test = zeros(size(test_X, 1), size(csp_filter', 1));
    for epoch = 1:size(test_X, 1)
        for filter = 1:size(csp_filter', 1)
            csp_test(epoch, filter) = log(csp_filter(:, filter)' * squeeze(test_X(epoch, :, :)) * ...
                                          squeeze(test_X(epoch, :, :))' * csp_filter(:, filter));
        end
    end
    
    % LDA
    Mdl = fitcdiscr(csp_train, train_y);
    test_y = predict(Mdl, csp_test);
    
end

