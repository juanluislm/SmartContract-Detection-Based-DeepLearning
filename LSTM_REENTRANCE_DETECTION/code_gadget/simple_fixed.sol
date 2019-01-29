mapping (address => uint) public credit;
function donate(address to) payable public
credit[to] += msg.value;
function withdraw(uint amount) public
if (credit[msg.sender]>= amount)
credit[msg.sender]-=amount;
require(msg.sender.call.value(amount)());
