function addToBalance() payable
userBalance[msg.sender] += msg.value;
function withdrawBalance()
if(!(msg.sender.call.value(userBalance[msg.sender])()))
userBalance[msg.sender] = 0;
function withdrawBalance_fixed()
uint amount = userBalance[msg.sender];
userBalance[msg.sender] = 0;
if(!(msg.sender.call.value(amount)()))
