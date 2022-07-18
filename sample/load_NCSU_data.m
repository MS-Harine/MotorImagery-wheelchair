function data = load_NCSU_data()

    % Trigger information
    %========================
    TRIG_NAME = ["CROSS", "FEEDBACK", "LEFT", "RIGHT", "UP", "DOWN"];
    TRIG_SIGN = [786, 781, 769, 770, 780, 774];
    %========================

    default_dir = fileparts(mfilename('fullpath'));
    data_list = dir(fullfile(default_dir, 'data', 'sub*', 'sess*', '*.gdf'));
    data = struct();
    
    for i = 1:length(data_list)
        current_data = pop_biosig(fullfile(data_list(i).folder, data_list(i).name));
        data(i).data = current_data.data;
        
        if strcmp(data_list(i).name, 'Resting.gdf')
            data(i).event = [];
            data(i).run = 'Resting';
        else
            current_data.event(~cellfun(@isTrigger, {current_data.event.edftype})) = [];
            trigger = struct('type', string([current_data.event.edftype]'), 'latency', [current_data.event.latency]');
            for trig = 1:length(TRIG_NAME)
                trigger.type = strrep(trigger.type, string(TRIG_SIGN(trig)), TRIG_NAME(trig));
            end
            data(i).event = table2struct(struct2table(trigger));
            data(i).run = str2double(regexp(data_list(i).name, '\d*', 'match')');
        end
        
        data(i).srate = current_data.srate;
        [directory, session] = fileparts(data_list(i).folder);
        [~, subject] = fileparts(directory);
        data(i).subject = str2double(regexp(subject, '\d*', 'match')');
        data(i).session = str2double(regexp(session, '\d*', 'match')');
        data(i).filename = data_list(i).name;
        data(i).chanlocs = current_data.chanlocs;
    end
    
    function result = isTrigger(value)
        if isempty(value)
            result = false;
        else
            result = any(value == TRIG_SIGN);
        end
    end
end