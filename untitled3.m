% Create a GUI with a button
fig = figure;
quitFlag=0
quitexecutionButton = uicontrol('Style', 'pushbutton', 'String', 'Quit Execution', 'Callback', @quitButtonPressed);

% Callback function for the button press event


% Your main code or loop
while true
    quitFlag
    quitButtonPressed
    % Your main code logic goes here
    quitexecutionButton
    pause(1)
    % Check if the quit flag is set, and break the loop if true
    % if quitFlag
    %     disp('User wants to quit execution');
    %     break;
    % end
end

function quitFlag = quitButtonPressed()
    disp('User pressed quitexecutionButton');
    quitFlag = 1;
    % Add your code here to handle the button press
    % For example, set a flag or perform specific actions
end
