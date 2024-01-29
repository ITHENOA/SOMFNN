function choosetype = myFunction(type)
% List of available type strings
available_types = {'int', 'double', 'single', 'char', 'logical'};

% Prompt the user to enter a value for 'type'
type = inputdlg('Enter a value for type', 'Type Selection', 1, available_types);

% Check if the input type is in the list of available types
if any(strcmp(type, available_types))
    % Do something with the input type
else
    % If the input type is not in the list of available types, suggest the available types
    uiwait(msgbox(sprintf('The input type is not valid. Please choose one of the following types:\n%s', strjoin(available_types, '\n')), 'Invalid Type', 'error'));
end

end
