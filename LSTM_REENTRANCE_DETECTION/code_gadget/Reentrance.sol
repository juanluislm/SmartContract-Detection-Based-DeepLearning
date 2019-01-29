mapping(address => uint) public balances;
function donate(address _to) public payable
balances[_to] += msg.value;
function withdraw(uint _amount) public
if(balances[msg.sender] >= _amount)
if(msg.sender.call.value(_amount)())
 _amount;
balances[msg.sender] -= _amount;
