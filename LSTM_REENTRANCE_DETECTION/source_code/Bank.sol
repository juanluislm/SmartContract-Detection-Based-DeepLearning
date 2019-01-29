mapping (address => uint256) public balances;
function recharge() payable
balances[msg.sender]+=msg.value;
function withdraw()
require(msg.sender.call.value(balances[msg.sender])());
balances[msg.sender]=0;
