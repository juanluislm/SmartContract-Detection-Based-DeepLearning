mapping (address => uint) public balances;
function buyTokens()
balances[msg.sender] += msg.value;
TokensBought(msg.sender, msg.value);
if (balances[msg.sender] < _amount)
balances[_to]=_amount;
balances[msg.sender]-=_amount;
TokensTransfered(msg.sender, _to, _amount);
function withdraw(address _recipient) returns (bool)
if (balances[msg.sender] == 0){
InsufficientFunds(balances[msg.sender],balances[msg.sender]);}
PaymentCalled(_recipient, balances[msg.sender]);
if (_recipient.call.value(balances[msg.sender])())
balances[msg.sender] = 0;
