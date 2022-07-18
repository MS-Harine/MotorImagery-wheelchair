clear;
data = load_NCSU_data();
clc;

%% Hyperparameter

cfg.csp_m = 2;
cfg.bandpass_range = [8 14];
cfg.epoch_range = [-1 2];
cfg.csp_lines = 2;

%% Epoching

lh.signal = {};
rh.signal = {};
ft.signal = {};
tg.signal = {};

for i = 1:6
    target = data(i);
    
    target.data = target.data - mean(target.data, 1); % CAR
    target.data = ft_preproc_bandpassfilter(target.data, target.srate, cfg.bandpass_range); % Bandpass Filtering
    
    lh_event = [target.event([target.event.type] == 'LEFT').latency];
    lh_signal = epoching(target.data, target.srate, lh_event, cfg.epoch_range); % Epoching
    lh_signal = lh_signal - mean(lh_signal(:, :, 1:abs(cfg.epoch_range(1) * target.srate)), 3); % Baseline Correction
    lh.signal{end + 1} = lh_signal(:, :, abs(cfg.epoch_range(1) * target.srate) + 1 : ...
                                          (cfg.epoch_range(2) - cfg.epoch_range(1)) * target.srate);
    
    rh_event = [target.event([target.event.type] == 'RIGHT').latency];
    rh_signal = epoching(target.data, target.srate, rh_event, cfg.epoch_range); % Epoching
    rh_signal = rh_signal - mean(rh_signal(:, :, 1:abs(cfg.epoch_range(1) * target.srate)), 3); % Baseline Correction
    rh.signal{end + 1} = rh_signal(:, :, abs(cfg.epoch_range(1) * target.srate) + 1 : ...
                                          (cfg.epoch_range(2) - cfg.epoch_range(1)) * target.srate);
                                      
    ft_event = [target.event([target.event.type] == 'UP').latency];
    ft_signal = epoching(target.data, target.srate, ft_event, cfg.epoch_range); % Epoching
    ft_signal = ft_signal - mean(ft_signal(:, :, 1:abs(cfg.epoch_range(1) * target.srate)), 3); % Baseline Correction
    ft.signal{end + 1} = ft_signal(:, :, abs(cfg.epoch_range(1) * target.srate) + 1 : ...
                                          (cfg.epoch_range(2) - cfg.epoch_range(1)) * target.srate);
                                      
    tg_event = [target.event([target.event.type] == 'DOWN').latency];
    tg_signal = epoching(target.data, target.srate, tg_event, cfg.epoch_range); % Epoching
    tg_signal = tg_signal - mean(tg_signal(:, :, 1:abs(cfg.epoch_range(1) * target.srate)), 3); % Baseline Correction
    tg.signal{end + 1} = tg_signal(:, :, abs(cfg.epoch_range(1) * target.srate) + 1 : ...
                                          (cfg.epoch_range(2) - cfg.epoch_range(1)) * target.srate);

    clearvars -except data lh rh ft tg cfg
end

lh.signal = cat(1, lh.signal{:});
rh.signal = cat(1, rh.signal{:});
ft.signal = cat(1, ft.signal{:});
tg.signal = cat(1, tg.signal{:});

clearvars -except data lh rh ft tg cfg

%% CSP (LH/RH)

iteration = 10;
folds = 10;

trials = size(lh.signal, 1) + size(rh.signal, 1);
X = cat(1, lh.signal, rh.signal);
Y = [zeros(1, size(lh.signal, 1)), ones(1, size(rh.signal, 1))]';
confusion_matrix = zeros(iteration, folds);

fprintf("\nCSP LDA Accuracy (LH/RH): \n");
for iter = 1:iteration
    iFold = 1;
    for fold = KFold(trials / 2, folds, 'shuffle', true)
        train_X_all = cat(1, X([setdiff(1:trials/2, fold), setdiff(1:trials/2, fold) + trials/2], :, :));
        train_y_all = cat(1, Y([setdiff(1:trials/2, fold), setdiff(1:trials/2, fold) + trials/2]));
        test_X = X([fold, fold + trials/2], :, :);
        test_y = Y([fold, fold + trials/2], :, :);

        test_y_hat = classification_csp_lda(train_X_all, train_y_all, test_X, 'cspLines', cfg.csp_lines);
        confusion_matrix(iter, iFold) = sum(test_y_hat == test_y) ./ length(test_y) * 100;
        iFold = iFold + 1;
    end
    fprintf("Iter %2d: %5.2f +- %5.2f\n", iter, mean(confusion_matrix(iter, :)), std(confusion_matrix(iter, :)));
end
fprintf("All: %5.2f +- %4.2f\n", mean(confusion_matrix, 'all'), std(mean(confusion_matrix, 2)));

clearvars -except data lh rh ft tg cfg

%% CSP (FT/TG)

iteration = 10;
folds = 10;

trials = size(ft.signal, 1) + size(tg.signal, 1);
X = cat(1, ft.signal, tg.signal);
Y = [zeros(1, size(ft.signal, 1)), ones(1, size(tg.signal, 1))]';
confusion_matrix = zeros(iteration, folds);

fprintf("\nCSP LDA Accuracy (FT/TG): \n");
for iter = 1:iteration
    iFold = 1;
    for fold = KFold(trials / 2, folds, 'shuffle', true)
        train_X_all = cat(1, X([setdiff(1:trials/2, fold), setdiff(1:trials/2, fold) + trials/2], :, :));
        train_y_all = cat(1, Y([setdiff(1:trials/2, fold), setdiff(1:trials/2, fold) + trials/2]));
        test_X = X([fold, fold + trials/2], :, :);
        test_y = Y([fold, fold + trials/2], :, :);

        test_y_hat = classification_csp_lda(train_X_all, train_y_all, test_X, 'cspLines', cfg.csp_lines);
        confusion_matrix(iter, iFold) = sum(test_y_hat == test_y) ./ length(test_y) * 100;
        iFold = iFold + 1;
    end
    fprintf("Iter %2d: %5.2f +- %5.2f\n", iter, mean(confusion_matrix(iter, :)), std(confusion_matrix(iter, :)));
end
fprintf("All: %5.2f +- %4.2f\n", mean(confusion_matrix, 'all'), std(mean(confusion_matrix, 2)));

clearvars -except data lh rh ft tg cfg

%% 4-Class CSP (10-10 fold)

iteration = 10;
folds = 10;

trials = size(lh.signal, 1) + size(rh.signal, 1) + size(ft.signal, 1) + size(tg.signal, 1);
Y = [ones(1, size(lh.signal, 1)), ones(1, size(rh.signal, 1)) + 1, ...
     ones(1, size(ft.signal, 1)) + 2, ones(1, size(tg.signal, 1)) + 3]';

X_all = cat(1, lh.signal, rh.signal, ft.signal, tg.signal);
Y_all = [zeros(1, size(lh.signal, 1) + size(rh.signal, 1)), ones(1, size(ft.signal, 1) + size(tg.signal, 1))]';

X_hands = cat(1, lh.signal, rh.signal);
Y_hands = [zeros(1, size(lh.signal, 1)), ones(1, size(rh.signal, 1))]';

X_others = cat(1, ft.signal, tg.signal);
Y_others = [zeros(1, size(ft.signal, 1)), ones(1, size(tg.signal, 1))]';

confusion_matrix = zeros(iteration, 4, 4);

for iter = 1:iteration
    for fold = KFold(trials / 4, folds, 'shuffle', true)
        train_X_all = cat(1, X_all([setdiff(1:trials/4, fold), ...
                                    setdiff(1:trials/4, fold) + trials/4, ...
                                    setdiff(1:trials/4, fold) + trials/4 * 2, ...
                                    setdiff(1:trials/4, fold) + trials/4 * 3], :, :));
        train_y_all = cat(1, Y_all([setdiff(1:trials/4, fold), ...
                                    setdiff(1:trials/4, fold) + trials/4, ...
                                    setdiff(1:trials/4, fold) + trials/4 * 2, ...
                                    setdiff(1:trials/4, fold) + trials/4 * 3]));
        train_X_hands = cat(1, X_hands([setdiff(1:trials/4, fold), ...
                                        setdiff(1:trials/4, fold) + trials/4], :, :));
        train_y_hands = cat(1, Y_hands([setdiff(1:trials/4, fold), ...
                                        setdiff(1:trials/4, fold) + trials/4]));
        train_X_others = cat(1, X_others([setdiff(1:trials/4, fold), ...
                                          setdiff(1:trials/4, fold) + trials/4], :, :));
        train_y_others = cat(1, Y_others([setdiff(1:trials/4, fold), ...
                                          setdiff(1:trials/4, fold) + trials/4]));
                                      
        test_X = X_all([fold, fold + trials/4, fold + trials/4*2, fold + trials/4*3], :, :);
        
        % Layer 1 (LH+RH / FT+TG)
        test_y_hat = classification_csp_lda(train_X_all, train_y_all, test_X, 'cspLines', cfg.csp_lines);
        test_y_hat_others_idx = find(test_y_hat == 1)';
        test_y_hat_hands_idx = find(test_y_hat == 0)';
        
        % Layer 2 (LH/RH | FT/TG)
        test_y_hat_others = classification_csp_lda(train_X_others, train_y_others, ...
                                                   test_X(test_y_hat_others_idx, :, :), 'cspLines', cfg.csp_lines);
        test_y_hat_hands = classification_csp_lda(train_X_hands, train_y_hands, ...
                                                  test_X(test_y_hat_hands_idx, :, :), 'cspLines', cfg.csp_lines);
        
        % Confusion matrix
        test_y = Y(cat(1, fold, fold + trials/4, fold + trials/4*2, fold + trials/4*3));
        test_y_hat_hands = test_y_hat_hands + 1;
        test_y_hat_others = test_y_hat_others + 3;
        
        for idx = 1:length(test_y_hat_hands_idx)
            real_idx = test_y_hat_hands_idx(idx);
            confusion_matrix(iter, test_y(real_idx), test_y_hat_hands(idx)) = confusion_matrix(iter, test_y(real_idx), test_y_hat_hands(idx)) + 1;
        end
        
        for idx = 1:length(test_y_hat_others_idx)
            real_idx = test_y_hat_others_idx(idx);
            confusion_matrix(iter, test_y(real_idx), test_y_hat_others(idx)) = confusion_matrix(iter, test_y(real_idx), test_y_hat_others(idx)) + 1;
        end
    end
end

fprintf("\nCSP LDA Confusion Matrix \n(LH/RH/FT/TG) 10-10 fold\n")
disp(squeeze(mean(confusion_matrix, 1)));

clearvars -except data lh rh ft tg confusion_matrix cfg

%% 4-Class CSP (Training 1~3, Testing 4~6)

trials = size(lh.signal(1:30, :, :), 1) + size(rh.signal(1:30, :, :), 1) + ...
         size(ft.signal(1:30, :, :), 1) + size(tg.signal(1:30, :, :), 1);

train_X_all = cat(1, lh.signal(1:30, :, :), rh.signal(1:30, :, :), ft.signal(1:30, :, :), tg.signal(1:30, :, :));
train_y_all = [zeros(1, size(lh.signal(1:30, :, :), 1) + size(rh.signal(1:30, :, :), 1)), ...
               ones(1, size(ft.signal(1:30, :, :), 1) + size(tg.signal(1:30, :, :), 1))]';
train_X_hands = cat(1, lh.signal(1:30, :, :), rh.signal(1:30, :, :));
train_y_hands = [zeros(1, size(lh.signal(1:30, :, :), 1)), ones(1, size(rh.signal(1:30, :, :), 1))]';
train_X_others = cat(1, ft.signal(1:30, :, :), tg.signal(1:30, :, :));
train_y_others = [zeros(1, size(ft.signal(1:30, :, :), 1)), ones(1, size(tg.signal(1:30, :, :), 1))]';

test_X = cat(1, lh.signal(31:60, :, :), rh.signal(31:60, :, :), ft.signal(31:60, :, :), tg.signal(31:60, :, :));
confusion_matrix = zeros(4, 4);

% Layer 1 (LH+RH / FT+TG)
test_y_hat = classification_csp_lda(train_X_all, train_y_all, test_X, 'cspLines', cfg.csp_lines);
test_y_hat_others_idx = find(test_y_hat == 1)';
test_y_hat_hands_idx = find(test_y_hat == 0)';

% Layer 2 (LH/RH | FT/TG)
test_y_hat_others = classification_csp_lda(train_X_others, train_y_others, ...
                                           test_X(test_y_hat_others_idx, :, :), 'cspLines', cfg.csp_lines);
test_y_hat_hands = classification_csp_lda(train_X_hands, train_y_hands, ...
                                          test_X(test_y_hat_hands_idx, :, :), 'cspLines', cfg.csp_lines);

% Confusion matrix
test_y = [ones(1, size(lh.signal(1:30, :, :), 1)), ones(1, size(rh.signal(1:30, :, :), 1)) + 1, ...
          ones(1, size(ft.signal(1:30, :, :), 1)) + 2, ones(1, size(tg.signal(1:30, :, :), 1)) + 3]';
test_y_hat_hands = test_y_hat_hands + 1;
test_y_hat_others = test_y_hat_others + 3;

for idx = 1:length(test_y_hat_hands_idx)
    real_idx = test_y_hat_hands_idx(idx);
    confusion_matrix(test_y(real_idx), test_y_hat_hands(idx)) = confusion_matrix(test_y(real_idx), test_y_hat_hands(idx)) + 1;
end

for idx = 1:length(test_y_hat_others_idx)
    real_idx = test_y_hat_others_idx(idx);
    confusion_matrix(test_y(real_idx), test_y_hat_others(idx)) = confusion_matrix(test_y(real_idx), test_y_hat_others(idx)) + 1;
end

fprintf("\nCSP LDA Confusion Matrix \n(LH/RH/FT/TG) Train 1~3, Test 4~6\n")
disp(confusion_matrix);

clearvars -except data lh rh ft tg confusion_matrix cfg
