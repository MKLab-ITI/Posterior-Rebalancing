function [A line_names]=import_output(file_name,start_line)
    if(nargin<2)
        start_line = 0;
    end
    A = [];
    line_names = {};
    current_line_id = 0;
    fid = fopen(file_name);
    %ignore first lines
    for i=1:start_line
        fgetl(fid);
    end
    %traverse file lines
    while(true)
        current_line_id = current_line_id+1;
        %read line
        line = fgetl(fid);
        if(~ischar(line))
            break
        end
        %disp(line)%show file line
        %remove useless characters
        line = strrep(line, '\', '');
        line = strrep(line, '%', '');
        line = strrep(line, '\textbf{', '');
        line = strrep(line, '}', '');
        line_content = textscan(line,'%s%d%d%d%d%d%d%d%d%d%d%d%d','Delimiter','&');
        line_names{current_line_id} = line_content{1};
        for i=2:length(line_content)
            A(current_line_id,i-1) = line_content{i}; 
        end
    end
    fclose(fid);
end