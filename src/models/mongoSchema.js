const mongoose = require("mongoose")
const validator=require("validator")
const bcrypt=require("bcryptjs")


const userSchema = mongoose.Schema({

    firstname:{
        type:String,
        required:true,
        trim:true
    },
    lastname:{
        type:String,
        required:true,
        trim:true
    },
    email:{
        type:String,
        unique: true,
        required:true,
        trim:true,
        lowercase:true,
        validate(value){
            if(!validator.isEmail(value)){
                throw new Error("Email is invalid!")
            }
        }
    },
    password:{
        type:String,
        required:true,
        trim:true,
            
    }
})


//userdef function for hiding private data
userSchema.methods.toJSON = function(){
    const user = this
    const userObj = user.toObject()
    delete userObj.password
    return userObj
} 

//using mongoose middleware for hashing passwords
userSchema.pre("save",async function (next) {
    const user =this
    console.log("user data received")
   if(user.isModified('password')){
       user.password=await bcrypt.hash(user.password,8)

   }
    next()
})

//userdef function for authentication
userSchema.statics.findByCredentials = async (email,password) => {
    const user = await User.findOne({ email })
    if(!user){
        throw new Error("Email is incorrect")
    }
    const isMatched = await bcrypt.compare(password,user.password)
    if(!isMatched){
        throw new Error("password is incorrect")
    }
    return user
}

//creating a user model
const User = mongoose.model('userAccounts',userSchema)

module.exports=User