import React from 'react'
import './Sidebar.css'
import { assets } from '../../../public/assets/assets'
import { useState } from 'react'

function Sidebar() {
  const [extended, setExtended] = useState(false)


  return (
  <div className='sidebar'>
  <div className="top">
        <img onClick={()=>setExtended(prev=>!prev)} className='menu' src={assets.menu_icon} alt="" />
        <div className="new-chart">
          <img src={assets.plus_icon} alt="" />
          {extended ? <p>New Chart</p>: null}
        </div>
      </div>
      {extended?
            <div className="recent">
            <p className="recent-title">Recent</p>
            <div className="recent-entry">
              <img src={assets.message_icon} alt="" />
              <p>What is react ... </p>
            </div>
          </div> :
          null
       }
     <div>
      <div className="bottom">
        <div className="bottom-item recent-entry" >
          <img src={assets.question_icon} alt="" />
          {extended ? <p>Help</p>: null }
        </div>
        <div className="bottom-item recent-entry" >
          <img src={assets.history_icon} alt="" />
          {extended ? <p>activity</p> : null }
        </div>
        <div className="bottom-item recent-entry" >
          <img src={assets.setting_icon} alt="" />
          { extended ? <p>settings</p> : null }
        </div>
        </div>
        
        </div> 
    </div>

  )
}

export default Sidebar;



