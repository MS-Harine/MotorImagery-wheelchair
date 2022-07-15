function data = load_data()

    % Trigger information
    %========================
    TRIG_NAME = ["CROSS", "FEEDBACK", "LEFT", "RIGHT", "UP", "DOWN"];
    TRIG_SIGN = [786, 781, 769, 770, 780, 774];
    %========================

    current_dir = fileparts(mfilename('fullpath'));
    data_path = fullfile('data', 's01');
    file_list = dir(fullfile(current_dir, data_path, '**', '*.gdf'));

    data = struct([]);
    for idx = 1:length(file_list)
        file = file_list(idx);
        
        [~, session] = fileparts(file.folder);
        session = str2double(regexp(session, '\d*', 'match')');
        
        filename = fullfile(file.folder, file.name);
        EEG = pop_biosig(filename);
        EEG.event(~cellfun(@isTrigger, {EEG.event.edftype})) = [];
        
        trigger = struct('type', string([EEG.event.edftype].'), 'latency', [EEG.event.latency].');
        for trig = 1:length(TRIG_NAME)
            trigger.type = strrep(trigger.type, string(TRIG_SIGN(trig)), TRIG_NAME(trig));
        end

        run.data = EEG.data;
        run.event = table2struct(struct2table(trigger));
        run.srate = EEG.srate;
        run.chanlocs = EEG.chanlocs;
        run.session = session;
        run.filename = file.name;
        
        if isempty(data)
            data = run;
        end
        data(idx) = run;
    end

    function result = isTrigger(value)
        if isempty(value)
            result = false;
        else
            result = any(value == TRIG_SIGN);
        end
    end

end

